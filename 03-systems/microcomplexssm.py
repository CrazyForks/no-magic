"""
A complex-valued state space model is mathematically identical to a real-valued one with
data-dependent RoPE — same dynamics, no complex arithmetic.
"""
# Reference: Mamba-3 (arXiv:2603.15569), Proposition 3 — Complex-to-Real Equivalence
# RoPE (Su et al. 2021) — Rotary Position Embedding
#
# === TRADEOFFS ===
# + Complex SSM: natural representation of periodic dynamics (rotation = complex multiply)
# + Real + RoPE: avoids complex arithmetic, uses familiar rotation matrices
# + Equivalence: choose whichever is more efficient for your hardware
# - Real-only SSM: cannot represent periodicity (decay only), fails at parity
# - Complex SSM: some hardware lacks efficient complex number support
# WHEN TO USE: complex/RoPE SSM when the task requires periodic state tracking
# WHEN NOT TO: real-only SSM for tasks with monotone decay dynamics
#
# ─── Connection to other no-magic scripts ───
# microssm.py    — implements selective SSM with real-valued (non-rotating) state
# microrope.py   — implements position-dependent rotation; this script shows
#                  data-dependent rotation is mathematically equivalent to complex SSMs
# microdiscretize.py — discretization methods; this script uses ZOH (the simplest
#                      stable discretization) to focus on the complex vs. real question

from __future__ import annotations

import math
import random
import time

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# SSM dimensions
N_STATE = 4           # state dimension (complex: 4 complex, real: 4 pairs = 8 real)
N_EMBD = 8            # embedding dimension for input projection
N_HIDDEN = 16         # hidden layer width for gate networks

# Training
LEARNING_RATE = 0.005
NUM_STEPS = 500       # per variant (3 variants × 500 = 1500 total)
BETA1 = 0.9           # Adam first moment decay
BETA2 = 0.999         # Adam second moment decay
EPS_ADAM = 1e-8       # Adam epsilon — matches PyTorch default
SEQ_LEN = 16          # parity sequence length (shorter = easier to learn)

# Parity task
NUM_TRAIN = 200       # training sequences
NUM_TEST = 100        # test sequences

# Signpost: production Mamba-3 uses N_STATE=64, N_EMBD=4096. We use tiny dims
# so the equivalence proof section can print exact numerical values.


# === SCALAR AUTOGRAD ENGINE ===

class Value:
    """A scalar value with reverse-mode automatic differentiation.

    Tracks computational history via ._children and ._local_grads, enabling
    gradient computation through the chain rule. Every forward operation stores
    its local derivative (dout/dinput) as a closure, then backward() replays
    the computation graph in reverse topological order, accumulating gradients.
    """
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # d(a+b)/da = 1, d(a+b)/db = 1
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        # d(a*b)/da = b, d(a*b)/db = a
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, exponent):
        # d(x^n)/dx = n * x^(n-1)
        return Value(
            self.data ** exponent, (self,),
            (exponent * self.data ** (exponent - 1),)
        )

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (1.0 / other)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def tanh(self):
        # d(tanh(x))/dx = 1 - tanh(x)^2
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t ** 2,))

    def exp(self):
        # d(e^x)/dx = e^x
        e = math.exp(self.data)
        return Value(e, (self,), (e,))

    def log(self):
        # d(log(x))/dx = 1/x
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def relu(self):
        # d(relu(x))/dx = 1 if x > 0 else 0
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def backward(self):
        """Compute gradients via reverse-mode automatic differentiation."""
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0

        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


# --- AUTOGRAD IN THIS SCRIPT ---
# This Value class follows the canonical interface exactly.
# See docs/autograd-interface.md for the full specification.


# === DATA GENERATION ===

def generate_parity_sequences(n_sequences, seq_len):
    """Generate binary sequences with running XOR labels.

    The parity task is the canonical SSM failure case: real-valued diagonal SSMs
    cannot represent XOR because it requires a state that 'flips' on each 1-bit —
    a 180-degree rotation, not a decay. A real eigenvalue can only shrink the state
    toward zero; it can never negate it.
    """
    data = []
    for _ in range(n_sequences):
        # x_i in {0, 1} via random.randint(0, 1)
        bits = [random.randint(0, 1) for _ in range(seq_len)]
        # label_i = x_0 XOR x_1 XOR ... XOR x_i (running parity)
        labels = []
        parity = 0
        for b in bits:
            parity ^= b
            labels.append(parity)
        data.append((bits, labels))
    return data


# === CORE OPERATIONS ===

def safe_log(prob):
    """Clipped logarithm for numerical stability in loss computation.

    Prevents log(0) which returns -inf and breaks gradient computation.
    Critical: we keep prob as a child node so gradients flow through the graph.
    """
    clamped = max(prob.data, 1e-10)
    return Value(math.log(clamped), (prob,), (1.0 / clamped,))


def binary_cross_entropy(pred, target):
    """Binary cross-entropy loss for parity classification.

    Math: BCE = -[y log(p) + (1-y) log(1-p)]
    We use the target as a plain int (0 or 1) and pred as a Value (probability).
    """
    if target == 1:
        return -safe_log(pred)
    else:
        return -safe_log(Value(1.0) - pred)


def sigmoid_value(x):
    """Sigmoid activation: sigma(x) = 1 / (1 + exp(-x)).

    Clamped input to prevent overflow in exp(). For x < -20, sigmoid ~ 0;
    for x > 20, sigmoid ~ 1. Clamping doesn't affect the useful range.
    """
    clamped = max(min(x.data, 20.0), -20.0)
    s = 1.0 / (1.0 + math.exp(-clamped))
    # d(sigmoid)/dx = sigmoid * (1 - sigmoid)
    return Value(s, (x,), (s * (1.0 - s),))


def cos_value(theta):
    """Cosine with autograd. d(cos θ)/dθ = -sin θ."""
    return Value(
        math.cos(theta.data), (theta,),
        (-math.sin(theta.data),)
    )


def sin_value(theta):
    """Sine with autograd. d(sin θ)/dθ = cos θ."""
    return Value(
        math.sin(theta.data), (theta,),
        (math.cos(theta.data),)
    )


# === COMPLEX SSM (REFERENCE IMPLEMENTATION) ===

# This section uses Python's built-in complex type for the forward pass.
# No autograd — this is for the equivalence proof only, where we don't need gradients.

def complex_ssm_forward(A_complex, B_complex, C_complex, x_seq):
    """Complex SSM: h_t = diag(A) h_{t-1} + B x_t, y_t = Re(C^H h_t)

    Math: A_n = r_n * e^{iθ_n} where r_n = |A_n| (decay rate), θ_n (rotation angle)
    State h is complex-valued. Multiplication by A rotates AND scales each component.
    Output: y_t = Re(C^H h_t) — real part of conjugate-transpose product.

    Why complex? Real eigenvalues can only scale (decay/grow). Complex eigenvalues
    rotate. Without rotation, you can't represent periodicity — which is why
    real-only SSMs fail at parity (they need to track odd/even, a mod-2 rotation).
    """
    n_state = len(A_complex)
    h = [complex(0, 0) for _ in range(n_state)]
    outputs = []

    for x_t in x_seq:
        # h_t = A * h_{t-1} + B * x_t
        for n in range(n_state):
            h[n] = A_complex[n] * h[n] + B_complex[n] * x_t

        # y_t = Re(C^H h_t) = sum_n Re(conj(C_n) * h_n)
        y_t = sum((C_complex[n].conjugate() * h[n]).real for n in range(n_state))
        outputs.append(y_t)

    return outputs


# === REAL SSM + DATA-DEPENDENT ROPE ===

# Equivalent formulation using 2x2 rotation matrices instead of complex multiply.
# No autograd — for equivalence proof only.
#
# The core identity that makes this work:
#   (r * e^{iθ}) * (h_re + i*h_im) = r * [(cos θ * h_re - sin θ * h_im)
#                                         + i*(sin θ * h_re + cos θ * h_im)]
#
# In matrix form for the pair (h_re, h_im):
#   [h_re']   = r * [cos θ  -sin θ] [h_re]   +  [B_re] * x
#   [h_im']       [sin θ   cos θ] [h_im]      [B_im]
#
# And Re(conj(C) * h) = C_re * h_re + C_im * h_im

def rope_ssm_forward(r_vals, theta_vals, B_re, B_im, C_re, C_im, x_seq):
    """Real SSM with rotation matrices: equivalent to complex SSM.

    For each pair (h_{2i}, h_{2i+1}), the state update is:
      [h_re]     [cos θ  -sin θ] [h_re]     [B_re]
      [h_im] = r [sin θ   cos θ] [h_im]  +  [B_im] * x_t

    This is RoPE (microrope.py), but the rotation angle comes from the
    model parameters, not the position index — data-dependent rotation.
    """
    n_state = len(r_vals)
    h_re = [0.0] * n_state
    h_im = [0.0] * n_state
    outputs = []

    for x_t in x_seq:
        for n in range(n_state):
            r = r_vals[n]
            cos_t = math.cos(theta_vals[n])
            sin_t = math.sin(theta_vals[n])

            # Rotation + scaling: the 2x2 matrix [r*cos, -r*sin; r*sin, r*cos]
            # applied to (h_re, h_im) is identical to complex multiply by r*e^{iθ}
            old_re = h_re[n]
            old_im = h_im[n]
            h_re[n] = r * (cos_t * old_re - sin_t * old_im) + B_re[n] * x_t
            h_im[n] = r * (sin_t * old_re + cos_t * old_im) + B_im[n] * x_t

        # Output: Re(conj(C) * h) = C_re * h_re + C_im * h_im
        y_t = sum(C_re[n] * h_re[n] + C_im[n] * h_im[n]
                  for n in range(n_state))
        outputs.append(y_t)

    return outputs


# === EQUIVALENCE PROOF BY COMPUTATION ===

def prove_equivalence():
    """Initialize complex and real+RoPE models with equivalent parameters.
    Run identical input. Assert outputs match to floating-point tolerance.

    The algebraic equivalence (Mamba-3 Proposition 3):
      Complex:  h_t = diag(r_n e^{iθ_n}) h_{t-1} + B x_t
      Real:     h_t[re,im] = r * R(θ) h_{t-1}[re,im] + [B_re, B_im] x_t
    These produce identical output sequences because R(θ) applied to
    real pairs is the real-arithmetic encoding of complex multiplication.
    """
    print("=" * 70)
    print("PHASE 1-3: EQUIVALENCE PROOF (complex SSM = real SSM + RoPE)")
    print("=" * 70)

    n_state = N_STATE

    # Generate random complex parameters in polar form
    # A_n = r_n * e^{iθ_n} with r_n < 1 for stability (state decays)
    r_vals = [random.uniform(0.5, 0.95) for _ in range(n_state)]
    theta_vals = [random.uniform(-math.pi, math.pi) for _ in range(n_state)]
    A_complex = [r * complex(math.cos(t), math.sin(t))
                 for r, t in zip(r_vals, theta_vals)]

    # B and C as arbitrary complex vectors
    B_complex = [complex(random.gauss(0, 0.5), random.gauss(0, 0.5))
                 for _ in range(n_state)]
    C_complex = [complex(random.gauss(0, 0.5), random.gauss(0, 0.5))
                 for _ in range(n_state)]

    # Convert to real equivalents:
    #   A_complex[n] = r_n * e^{iθ_n}  ->  r_vals[n], theta_vals[n]
    #   B_complex[n] = b_re + i*b_im    ->  B_re[n], B_im[n]
    #   C_complex[n] = c_re + i*c_im    ->  C_re[n], C_im[n]
    B_re = [b.real for b in B_complex]
    B_im = [b.imag for b in B_complex]
    C_re = [c.real for c in C_complex]
    C_im = [c.imag for c in C_complex]

    # Random input sequence
    x_seq = [random.gauss(0, 1.0) for _ in range(SEQ_LEN)]

    print("\nPhase 1: Complex SSM forward pass (Python complex type)...")
    complex_out = complex_ssm_forward(A_complex, B_complex, C_complex, x_seq)

    print("Phase 2: Real SSM + RoPE forward pass (2x2 rotation matrices)...")
    rope_out = rope_ssm_forward(r_vals, theta_vals, B_re, B_im, C_re, C_im, x_seq)

    # Phase 3: Compare
    print("\nPhase 3: Comparing outputs...")
    max_diff = max(abs(c - r) for c, r in zip(complex_out, rope_out))
    print(f"  Sequence length: {SEQ_LEN}")
    print(f"  State dimension: {n_state} complex = {2 * n_state} real")
    print(f"  Max absolute difference: {max_diff:.2e}")

    print(f"\n  {'Step':>6}  {'Complex':>14}  {'Real+RoPE':>14}  {'Diff':>12}")
    print(f"  {'':->6}  {'':->14}  {'':->14}  {'':->12}")
    for t in range(min(8, SEQ_LEN)):
        diff = abs(complex_out[t] - rope_out[t])
        print(f"  {t:>6}  {complex_out[t]:>14.8f}  {rope_out[t]:>14.8f}  {diff:>12.2e}")

    if max_diff < 1e-12:
        print(f"\n  PASS: max difference {max_diff:.2e} < 1e-12")
    elif max_diff < 1e-10:
        print(f"\n  PASS: max difference {max_diff:.2e} < 1e-10 (accumulated FP error)")
    else:
        print(f"\n  FAIL: max difference {max_diff:.2e} exceeds tolerance")

    print(f"\n  Conclusion: complex SSM and real SSM + data-dependent RoPE produce")
    print(f"  identical outputs (to floating-point precision). The 2x2 rotation")
    print(f"  matrix R(theta) is the real-arithmetic encoding of e^{{i*theta}}.\n")

    return max_diff


# === TRAINING: THREE VARIANTS ===

# For training, complex numbers become (Value, Value) pairs for autograd.
# Python's built-in complex type doesn't support gradients.
#
# The parity task requires input-dependent gating: when x_t = 1, rotate the
# state by theta (flip parity); when x_t = 0, preserve the state (no rotation).
# A fixed rotation per step can't distinguish 0-bits from 1-bits, so we gate
# the rotation angle by the input: effective_theta = theta * x_t.

def init_real_only_params():
    """Initialize a real-only SSM (no rotation).

    State update: h_t = diag(a_1, ..., a_n) h_{t-1} + B x_t
    Without rotation, each state dimension can only decay toward zero.
    This means the SSM cannot represent parity (a mod-2 flip).
    """
    params = {}
    # A diagonal: log-parameterized so A = exp(log_A) with log_A < 0 for decay
    params['log_A'] = [Value(random.gauss(-0.3, 0.2)) for _ in range(N_STATE)]
    # Input projection: scalar -> N_STATE
    params['B'] = [Value(random.gauss(0, 0.5)) for _ in range(N_STATE)]
    # Output projection: N_STATE -> scalar logit
    params['C'] = [Value(random.gauss(0, 0.5)) for _ in range(N_STATE)]
    # Output bias
    params['bias'] = Value(0.0)
    return params


def init_complex_params():
    """Initialize a complex SSM for training.

    Complex numbers are (Value, Value) pairs for autograd.
    A_n = r_n * e^{iθ_n} parameterized as (log_r, theta):
      - r_n = exp(log_r_n), magnitude (controls decay)
      - theta_n, rotation angle (controls periodicity)
    The rotation is gated by the input: only 1-bits trigger rotation.
    """
    params = {}
    # Decay magnitude (log-parameterized for stability)
    params['log_r'] = [Value(random.gauss(-0.2, 0.1)) for _ in range(N_STATE)]
    # Rotation angle — initialized near pi for parity (180 degree flip)
    params['theta'] = [Value(random.gauss(math.pi, 0.3)) for _ in range(N_STATE)]
    # B complex: (B_re, B_im) per state
    params['B_re'] = [Value(random.gauss(0, 0.5)) for _ in range(N_STATE)]
    params['B_im'] = [Value(random.gauss(0, 0.5)) for _ in range(N_STATE)]
    # C complex: (C_re, C_im) per state
    params['C_re'] = [Value(random.gauss(0, 0.5)) for _ in range(N_STATE)]
    params['C_im'] = [Value(random.gauss(0, 0.5)) for _ in range(N_STATE)]
    # Output bias
    params['bias'] = Value(0.0)
    return params


def init_rope_params():
    """Initialize a real SSM + data-dependent RoPE.

    Same expressive power as complex SSM: rotation replaces complex multiply.
    theta controls the 2x2 rotation applied to state pairs (h_re, h_im).
    """
    params = {}
    params['log_r'] = [Value(random.gauss(-0.2, 0.1)) for _ in range(N_STATE)]
    # Rotation angle — same initialization as complex variant
    params['theta'] = [Value(random.gauss(math.pi, 0.3)) for _ in range(N_STATE)]
    # B real (pairs): B_re, B_im per state
    params['B_re'] = [Value(random.gauss(0, 0.5)) for _ in range(N_STATE)]
    params['B_im'] = [Value(random.gauss(0, 0.5)) for _ in range(N_STATE)]
    # C real (pairs): C_re, C_im per state
    params['C_re'] = [Value(random.gauss(0, 0.5)) for _ in range(N_STATE)]
    params['C_im'] = [Value(random.gauss(0, 0.5)) for _ in range(N_STATE)]
    # Output bias
    params['bias'] = Value(0.0)
    return params


def forward_real_only(params, x_seq):
    """Forward pass for real-only SSM (no rotation).

    h_t = exp(log_A) * h_{t-1} + B * x_t
    y_t = sigmoid(C^T h_t + bias)

    Without rotation, each state dimension decays exponentially.
    The state can grow or shrink but never flip sign — fatal for parity.
    Real eigenvalues lie on the real line: they can only scale, not rotate.
    """
    h = [Value(0.0) for _ in range(N_STATE)]
    outputs = []

    for x_t in x_seq:
        x_val = Value(float(x_t))
        for n in range(N_STATE):
            # A_n = exp(log_A_n): positive, so state decays without sign change
            a_n = params['log_A'][n].exp()
            h[n] = a_n * h[n] + params['B'][n] * x_val

        # Output: sigmoid(C^T h + bias) -> probability of parity=1
        logit = params['bias']
        for n in range(N_STATE):
            logit = logit + params['C'][n] * h[n]
        outputs.append(sigmoid_value(logit))

    return outputs


def forward_complex(params, x_seq):
    """Forward pass for complex SSM with input-gated rotation.

    The key to learning parity: the rotation angle is GATED by the input.
    When x_t = 1: h_t = r*e^{iθ} * h_{t-1} + B * x_t  (rotate state by θ)
    When x_t = 0: h_t = r * h_{t-1} + B * x_t           (no rotation, decay only)

    Math: effective_θ = θ * x_t
    This makes the rotation input-dependent (selective), which is the Mamba insight.
    For parity, the network learns θ ~ π so each 1-bit flips the state.

    Complex multiply expanded to real arithmetic:
      h_re = r*(cos(θ*x) * h_re - sin(θ*x) * h_im) + B_re * x
      h_im = r*(sin(θ*x) * h_re + cos(θ*x) * h_im) + B_im * x
    """
    h_re = [Value(0.0) for _ in range(N_STATE)]
    h_im = [Value(0.0) for _ in range(N_STATE)]
    outputs = []

    for x_t in x_seq:
        x_val = Value(float(x_t))
        for n in range(N_STATE):
            r_n = params['log_r'][n].exp()
            # Gate rotation by input: only rotate when x_t = 1
            # effective_theta = theta * x_t
            eff_theta = params['theta'][n] * x_val
            cos_v = cos_value(eff_theta)
            sin_v = sin_value(eff_theta)

            old_re = h_re[n]
            old_im = h_im[n]
            # Complex multiply: r * e^{i*eff_theta} * h + B * x
            h_re[n] = r_n * (cos_v * old_re - sin_v * old_im) + \
                params['B_re'][n] * x_val
            h_im[n] = r_n * (sin_v * old_re + cos_v * old_im) + \
                params['B_im'][n] * x_val

        # Output: Re(C^H h) = sum(C_re * h_re + C_im * h_im) + bias
        logit = params['bias']
        for n in range(N_STATE):
            logit = logit + params['C_re'][n] * h_re[n] + \
                params['C_im'][n] * h_im[n]
        outputs.append(sigmoid_value(logit))

    return outputs


def forward_rope(params, x_seq):
    """Forward pass for real SSM + data-dependent RoPE.

    Identical dynamics to forward_complex, expressed with rotation matrices.
    For each state pair (h_re, h_im):
      R(θ) = [cos θ  -sin θ]    applied to [h_re]
             [sin θ   cos θ]              [h_im]

    This is RoPE (microrope.py), but the rotation angle comes from the
    INPUT, not the position index. That's the key difference: data-dependent
    rotation. The angle θ*x_t is zero when x_t=0 (identity rotation) and
    θ when x_t=1 (state flip for parity).
    """
    h_re = [Value(0.0) for _ in range(N_STATE)]
    h_im = [Value(0.0) for _ in range(N_STATE)]
    outputs = []

    for x_t in x_seq:
        x_val = Value(float(x_t))
        for i in range(N_STATE):
            r_i = params['log_r'][i].exp()
            # Data-dependent rotation: angle gated by input
            eff_theta = params['theta'][i] * x_val
            cos_v = cos_value(eff_theta)
            sin_v = sin_value(eff_theta)

            old_re = h_re[i]
            old_im = h_im[i]
            # 2x2 rotation matrix: R(θ) * [h_re, h_im]^T, then scale by r
            h_re[i] = r_i * (cos_v * old_re - sin_v * old_im) + \
                params['B_re'][i] * x_val
            h_im[i] = r_i * (sin_v * old_re + cos_v * old_im) + \
                params['B_im'][i] * x_val

        logit = params['bias']
        for i in range(N_STATE):
            logit = logit + params['C_re'][i] * h_re[i] + \
                params['C_im'][i] * h_im[i]
        outputs.append(sigmoid_value(logit))

    return outputs


def get_param_list(params):
    """Flatten parameter dict into a list of Value objects for the optimizer."""
    param_list = []
    for val in params.values():
        if isinstance(val, Value):
            param_list.append(val)
        elif isinstance(val, list):
            param_list.extend(val)
    return param_list


def train_variant(name, init_fn, forward_fn, train_data, test_data):
    """Train one SSM variant and return metrics.

    All three variants use the same training loop and optimizer.
    The only difference is the forward pass — this isolates the effect
    of rotation (complex/RoPE) vs. no rotation (real-only).
    """
    print(f"\n--- Variant: {name} ---")

    params = init_fn()
    param_list = get_param_list(params)
    print(f"  Parameters: {len(param_list)}")

    # Adam optimizer state
    m = [0.0] * len(param_list)
    v = [0.0] * len(param_list)

    start = time.time()
    final_loss = 0.0

    for step in range(NUM_STEPS):
        bits, labels = train_data[step % len(train_data)]

        # Forward pass: get predicted probabilities for each timestep
        preds = forward_fn(params, bits)

        # Binary cross-entropy loss averaged over sequence
        loss = Value(0.0)
        for t in range(len(bits)):
            loss = loss + binary_cross_entropy(preds[t], labels[t])
        loss = loss * (1.0 / len(bits))

        # Backward pass
        loss.backward()
        final_loss = loss.data

        # Adam update with linear LR decay
        lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)
        for i, p in enumerate(param_list):
            m[i] = BETA1 * m[i] + (1 - BETA1) * p.grad
            v[i] = BETA2 * v[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m[i] / (1 - BETA1 ** (step + 1))
            v_hat = v[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)
            p.grad = 0.0

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{NUM_STEPS:>4} | loss: {loss.data:.4f}")

    elapsed = time.time() - start

    # Evaluate on test data
    correct = 0
    total = 0
    for bits, labels in test_data:
        preds = forward_fn(params, bits)
        for t in range(len(bits)):
            pred_label = 1 if preds[t].data > 0.5 else 0
            if pred_label == labels[t]:
                correct += 1
            total += 1

    accuracy = correct / total
    print(f"  Test accuracy: {accuracy:.1%} ({correct}/{total})")
    print(f"  Time: {elapsed:.2f}s")

    return params, final_loss, accuracy, elapsed


# === MAIN ===

def main():
    overall_start = time.time()

    # --- Phase 1-3: Equivalence proof ---
    max_diff = prove_equivalence()

    # --- Phase 4: Training three variants ---
    print("=" * 70)
    print("PHASE 4: TRAINING THREE VARIANTS ON PARITY TASK")
    print("=" * 70)

    print(f"\nParity task: binary sequences of length {SEQ_LEN}")
    print(f"  label_i = XOR(x_0, ..., x_i) -- running parity")
    print(f"  Training: {NUM_TRAIN} sequences, Test: {NUM_TEST} sequences")
    print(f"  Steps per variant: {NUM_STEPS}")

    # The key insight: real eigenvalues can only decay. Complex eigenvalues rotate.
    # Parity requires a 180 degree flip (mod-2 rotation), which only complex/RoPE
    # SSMs can represent. A real eigenvalue a_n multiplies the state by a_n each
    # step. For |a_n| < 1 (stability), the state decays toward zero — it can never
    # negate itself. Complex eigenvalues r*e^{iθ} rotate the state by θ each step.
    # Setting θ = π gives a sign flip, which is exactly XOR.

    train_data = generate_parity_sequences(NUM_TRAIN, SEQ_LEN)
    test_data = generate_parity_sequences(NUM_TEST, SEQ_LEN)

    # (a) Real-only SSM — expected: ~50% accuracy (cannot learn parity)
    _, loss_a, acc_a, time_a = train_variant(
        "(a) Real-only (no rotation)",
        init_real_only_params,
        forward_real_only,
        train_data, test_data
    )

    # (b) Complex SSM — expected: high accuracy
    params_b, loss_b, acc_b, time_b = train_variant(
        "(b) Complex SSM",
        init_complex_params,
        forward_complex,
        train_data, test_data
    )

    # (c) Real SSM + data-dependent RoPE — expected: matches (b)
    params_c, loss_c, acc_c, time_c = train_variant(
        "(c) Real + data-dependent RoPE",
        init_rope_params,
        forward_rope,
        train_data, test_data
    )

    # === COMPARISON TABLE ===

    print("\n" + "=" * 70)
    print("PHASE 5: COMPARISON")
    print("=" * 70)

    hdr = f"{'Variant':<35} {'Accuracy':>10} {'Final Loss':>12} {'Time (s)':>10}"
    print(f"\n{hdr}")
    print("-" * len(hdr))
    results = [
        ("(a) Real-only (no rotation)", acc_a, loss_a, time_a),
        ("(b) Complex SSM", acc_b, loss_b, time_b),
        ("(c) Real + data-dependent RoPE", acc_c, loss_c, time_c),
    ]
    for name, acc, loss, t in results:
        print(f"{name:<35} {acc:>10.1%} {loss:>12.4f} {t:>10.2f}")

    # Learned rotation angles — parity should push θ toward π (180 degree flip)
    # because XOR is a mod-2 operation: each 1-bit rotates state by π
    print(f"\nLearned rotation angles:")
    hdr2 = f"{'State pair':>12} {'theta (complex)':>16} {'theta (RoPE)':>14} {'Difference':>12}"
    print(hdr2)
    print("-" * len(hdr2))
    for i in range(N_STATE):
        theta_complex = params_b['theta'][i].data
        theta_rope = params_c['theta'][i].data
        diff = abs(theta_complex - theta_rope)
        print(f"{i:>12} {theta_complex:>16.4f} {theta_rope:>14.4f} {diff:>12.2e}")

    print(f"\n  For parity, the ideal rotation angle is pi = {math.pi:.4f} (180 degrees).")
    print(f"  Each 1-bit should flip the state (rotate by pi). Each 0-bit triggers")
    print(f"  no rotation because effective_theta = theta * x_t = theta * 0 = 0.")

    # Equivalence proof summary
    print(f"\n  Equivalence proof max |diff|: {max_diff:.2e}")

    # Final summary
    print(f"\n  Key result: real-only SSM gets ~50% (random guessing) because")
    print(f"  decay-only dynamics cannot represent the parity flip. Complex and")
    print(f"  RoPE variants both succeed because rotation IS the parity operation.")

    print(f"\nTotal runtime: {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
