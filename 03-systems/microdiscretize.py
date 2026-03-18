"""
The choice of discretization method changes what a recurrence can learn — Euler, ZOH,
and trapezoidal aren't just approximations, they're different inductive biases.
"""
# Reference: Mamba-3 (arXiv:2603.15569), Section 3 — Discretization Methods
# S4 (Gu et al. 2022) — Bilinear/Tustin discretization for structured SSMs
#
# === TRADEOFFS ===
# + Trapezoidal: 2nd-order accuracy, implicit convolution replaces explicit short conv
# + ZOH: simple, stable for any delta, matches Mamba-1/2
# + Euler: cheapest computation (no exp), good intuition builder
# - Euler: conditionally stable — large delta causes divergence
# - Trapezoidal: requires storing x_{t-1} (size-2 convolution in recurrence)
# - ZOH: 1st-order only, holds input constant over the step
# WHEN TO USE: trapezoidal when accuracy matters and you can afford x_{t-1} storage
# WHEN NOT TO: Euler at production scale (use ZOH or trapezoidal)

# ─── Connection to other no-magic scripts ───
# microssm.py    — implements selective SSM with a single (ZOH) discretization;
#                  this script isolates discretization as the variable
# microrope.py   — position encoding via rotation; discretization choice affects
#                  how position information interacts with the state update

from __future__ import annotations

import math
import random
import time

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Continuous-time SSM dimensions
N_STATE = 4           # state dimension (diagonal A is N_STATE scalars)
N_INPUT = 1           # input dimension (scalar input for clarity)
N_OUTPUT = 1          # output dimension

# Training
LEARNING_RATE = 0.01  # Adam base learning rate
NUM_STEPS = 600       # training steps per method (3 methods × 600 = 1800 total)
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8       # matches PyTorch default; prevents division by zero in Adam

# Data generation
SEQ_LEN = 64          # sequence length for training
NUM_TRAIN = 200       # number of training sequences
NUM_TEST = 50         # number of test sequences

# Signpost: production SSMs use N_STATE=16-64 per channel with N_EMBD=768+.
# We use N_STATE=4, N_INPUT=1 to make the discretization math visible in output.


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

    # Arithmetic operations
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
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    # Activation functions
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
        """Compute gradients via reverse-mode automatic differentiation.

        Builds a topological ordering of the computation graph, then propagates
        gradients backward using the chain rule.
        """
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)

        # Seed: gradient of loss with respect to itself is 1
        self.grad = 1.0

        # Reverse topological order: gradients flow backward from output to inputs
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                # Chain rule: dLoss/dchild += dLoss/dv * dv/dchild
                child.grad += local_grad * v.grad


# --- AUTOGRAD IN THIS SCRIPT ---
# This Value class follows the canonical interface exactly.
# See docs/autograd-interface.md for the full specification.


# === DATA GENERATION ===

def generate_sine_data(n_sequences: int, seq_len: int):
    """Generate sin(t) sampled at non-uniform intervals.

    Non-uniform delta_t tests whether the discretization handles variable
    step sizes correctly. Euler drifts at large delta; ZOH/trapezoidal don't.
    The mix of clustered-small and occasional-large gaps stresses all three
    methods differently: Euler becomes unstable at large gaps, ZOH's
    "hold constant" assumption hurts when the signal changes fast between
    widely-spaced samples, and trapezoidal's interpolation of x_t and x_{t-1}
    gives it an edge on rapidly-varying signals.
    Returns list of (x_seq, delta_seq, target_seq) tuples.
    """
    data = []
    for _ in range(n_sequences):
        # Mixture of small and large gaps: draw from exponential distribution
        # then sort. This creates bursts of closely-spaced points with occasional
        # large jumps — realistic for irregularly-sampled time series.
        raw = []
        t = 0.0
        for _ in range(seq_len + 1):
            # Exponential gaps with mean 0.3, occasionally large
            gap = random.expovariate(3.0)  # mean=1/3
            t += gap
            raw.append(t)

        # Composite signal: sum of two sines at different frequencies.
        # Higher frequency means the signal changes faster between samples,
        # rewarding methods that interpolate (trapezoidal) over those that hold
        # constant (ZOH) or use a crude linear step (Euler).
        x_seq = [math.sin(raw[i]) + 0.5 * math.sin(3.0 * raw[i])
                 for i in range(seq_len)]
        delta_seq = [raw[i + 1] - raw[i] for i in range(seq_len)]
        target_seq = [math.sin(raw[i + 1]) + 0.5 * math.sin(3.0 * raw[i + 1])
                      for i in range(seq_len)]

        data.append((x_seq, delta_seq, target_seq))

    return data


def generate_parity_data(n_sequences: int, seq_len: int):
    """Generate binary sequences with running XOR labels.

    Parity is a stepping stone to full state-tracking. Without complex
    eigenvalues (that's microcomplexssm.py), this task is harder —
    real-valued SSMs must approximate the flip via decay dynamics.
    Returns list of (x_seq, delta_seq, target_seq) tuples.
    """
    data = []
    for _ in range(n_sequences):
        # x_i in {0, 1}, label_i = XOR(x_0, ..., x_i)
        x_seq = [random.randint(0, 1) for _ in range(seq_len)]
        # Uniform delta for parity — variable spacing isn't the point here
        delta_seq = [1.0] * seq_len

        running_xor = 0
        target_seq = []
        for x in x_seq:
            running_xor ^= x
            target_seq.append(float(running_xor))

        data.append((x_seq, delta_seq, target_seq))

    return data


# === CORE OPERATIONS ===

def mse_loss(predictions: list[Value], targets: list[float]) -> Value:
    """Mean squared error: (1/N) sum (pred_i - target_i)^2."""
    n = len(predictions)
    total = Value(0.0)
    for pred, tgt in zip(predictions, targets):
        diff = pred - tgt
        total = total + diff * diff
    return total * (1.0 / n)


def binary_cross_entropy_loss(
    predictions: list[Value], targets: list[float]
) -> Value:
    """BCE loss: -(1/N) sum [y*log(p) + (1-y)*log(1-p)].

    Uses sigmoid outputs and clipped log to prevent log(0). The gradient
    path flows through the Value graph because safe_log keeps the child link.
    """
    n = len(predictions)
    total = Value(0.0)
    for pred, tgt in zip(predictions, targets):
        # Clamp prediction to (1e-7, 1-1e-7) for numerical stability.
        # This prevents log(0) which returns -inf and breaks backprop.
        p_clamped = max(pred.data, 1e-7)
        p_clamped = min(p_clamped, 1.0 - 1e-7)
        log_p = Value(math.log(p_clamped), (pred,), (1.0 / p_clamped,))
        log_1mp = Value(
            math.log(1.0 - p_clamped), (pred,), (-1.0 / (1.0 - p_clamped),)
        )
        total = total + (-(tgt * log_p + (1.0 - tgt) * log_1mp))
    return total * (1.0 / n)


def sigmoid_value(x: Value) -> Value:
    """Sigmoid activation: 1 / (1 + exp(-x)).

    Used as the output nonlinearity for parity (binary classification).
    Clamped input to [-20, 20] to prevent overflow in exp().
    """
    clamped = max(min(x.data, 20.0), -20.0)
    s = 1.0 / (1.0 + math.exp(-clamped))
    # d(sigmoid)/dx = sigmoid * (1 - sigmoid)
    return Value(s, (x,), (s * (1.0 - s),))


# === CONTINUOUS-TIME SSM ===

# Math: h'(t) = A h(t) + B x(t)
#        y(t) = C h(t)
# A is diagonal (N_STATE scalars), parameterized as A_n = -exp(log_a_n)
# to guarantee A < 0 (stable decay). Same parameterization as microssm.py.

def init_ssm_params():
    """Initialize parameters for a continuous-time SSM.

    Returns a dict with:
    - log_A: log-parameterized diagonal A (N_STATE values, A_n = -exp(log_A_n))
    - B: input projection (N_STATE values)
    - C: output projection (N_STATE values)
    - D: skip connection scalar (feedthrough term)
    """
    params = {
        # A diagonal in log-space: A_n = -exp(log_A_n) ensures A < 0 (stable decay).
        # Initialized near -1 so exp(log_A) ~ exp(-1) ~ 0.37, giving moderate decay.
        'log_A': [Value(random.gauss(-1.0, 0.3)) for _ in range(N_STATE)],

        # B and C: input/output projections, small random init
        'B': [Value(random.gauss(0, 0.1)) for _ in range(N_STATE)],
        'C': [Value(random.gauss(0, 0.1)) for _ in range(N_STATE)],

        # D: skip connection — lets the model pass input directly to output.
        # Critical for the sine task where y ≈ x with a small phase shift.
        'D': [Value(0.0)],
    }
    return params


def collect_params(params: dict) -> list[Value]:
    """Flatten parameter dict into a list for optimizer bookkeeping."""
    flat = []
    for val in params.values():
        flat.extend(val)
    return flat


# === DISCRETIZATION METHODS ===

def euler_discretize(log_A: list[Value], B: list[Value], delta: Value):
    """First-order forward Euler discretization.

    Math: h_t = (I + delta * A) h_{t-1} + (delta * B) x_t
    This is the simplest approximation: replace the derivative with a finite
    difference. Conditionally stable: requires |1 + delta * a_n| < 1 for each
    eigenvalue a_n. For a_n = -0.1, stable when delta < 20.
    Production SSMs never use this — it's here as the baseline.
    """
    A_bar = []
    B_bar = []

    for n in range(N_STATE):
        # A_n = -exp(log_A_n), guaranteed negative
        neg_exp = -math.exp(log_A[n].data)
        A_n = Value(neg_exp, (log_A[n],), (neg_exp,))

        # A_bar_n = 1 + delta * A_n (Euler approximation of exp(delta * A_n))
        A_bar_n = Value(1.0) + delta * A_n
        A_bar.append(A_bar_n)

        # B_bar_n = delta * B_n
        B_bar_n = delta * B[n]
        B_bar.append(B_bar_n)

    return A_bar, B_bar


def zoh_discretize(log_A: list[Value], B: list[Value], delta: Value):
    """Zero-order hold discretization.

    Math: h_t = exp(delta * A) h_{t-1} + (exp(delta * A) - I) A^{-1} B x_t
    Assumes input held constant over [t, t+delta]. 1st-order accurate.
    Unconditionally stable: |exp(delta * a_n)| < 1 for all delta > 0 when a_n < 0.
    This is what Mamba-1/2 use.
    """
    A_bar = []
    B_bar = []

    for n in range(N_STATE):
        # A_n = -exp(log_A_n)
        neg_exp = -math.exp(log_A[n].data)
        A_n = Value(neg_exp, (log_A[n],), (neg_exp,))

        # A_bar_n = exp(delta * A_n) — the exact matrix exponential for diagonal A.
        # For scalar case: exp(delta * a_n) where a_n < 0, so 0 < A_bar_n < 1.
        delta_A = delta * A_n
        # Clamp to prevent underflow for very large negative values
        clamped_dA = max(delta_A.data, -20.0)
        exp_dA = math.exp(clamped_dA)
        A_bar_n = Value(exp_dA, (delta_A,), (exp_dA,))
        A_bar.append(A_bar_n)

        # B_bar_n = (exp(delta * A_n) - 1) / A_n * B_n
        # = (A_bar_n - 1) * A_n^{-1} * B_n
        # When A_n is very small, use L'Hopital: limit is delta * B_n.
        if abs(neg_exp) < 1e-12:
            B_bar_n = delta * B[n]
        else:
            # (exp(dA) - 1) / A_n is the exact integral of exp(A_n * s) from 0 to delta
            ratio = (exp_dA - 1.0) / neg_exp
            # Gradient of ratio w.r.t. delta_A: (delta_A * exp_dA - exp_dA + 1) / A_n^2
            # but we treat ratio as a function of delta_A for the chain rule
            B_bar_n = Value(ratio, (delta_A,), (
                (clamped_dA * exp_dA - exp_dA + 1.0) / (clamped_dA ** 2)
                if abs(clamped_dA) > 1e-8 else 0.5,
            )) * B[n]

        B_bar.append(B_bar_n)

    return A_bar, B_bar


def trapezoidal_discretize(
    log_A: list[Value], B: list[Value], delta: Value, alpha: float = 0.5
):
    """Generalized alpha-method on the exact exponential.

    Math: h_t = exp(delta * A) h_{t-1} + alpha * B_int * x_t
                                        + (1 - alpha) * B_int * x_{t-1}
      where B_int = (exp(delta * A) - I) A^{-1} B

    alpha=1.0 -> ZOH (all weight on x_t, recovers Mamba-1/2)
    alpha=0.5 -> trapezoidal (Mamba-3's exponential-trapezoidal)
    alpha=0.0 -> implicit (all weight on x_{t-1})

    The punchline: alpha=0.5 creates h_t = f(h_{t-1}, x_t, x_{t-1}) —
    a size-2 convolution in the recurrence. This is why Mamba-3 drops
    the explicit short convolution that Mamba-1/2 needed.

    NOTE: forward Euler (A_bar = I + delta * A) is a different, first-order
    approximation — it cannot be recovered from this exponential family.
    """
    A_bar = []
    B_bar_curr = []   # coefficient for x_t (weight = alpha)
    B_bar_prev = []   # coefficient for x_{t-1} (weight = 1 - alpha)

    for n in range(N_STATE):
        neg_exp = -math.exp(log_A[n].data)
        A_n = Value(neg_exp, (log_A[n],), (neg_exp,))

        # Same A_bar as ZOH: exp(delta * A_n)
        delta_A = delta * A_n
        clamped_dA = max(delta_A.data, -20.0)
        exp_dA = math.exp(clamped_dA)
        A_bar_n = Value(exp_dA, (delta_A,), (exp_dA,))
        A_bar.append(A_bar_n)

        # B_int = (exp(delta * A_n) - 1) / A_n * B_n — same integral as ZOH
        if abs(neg_exp) < 1e-12:
            B_int_n = delta * B[n]
        else:
            ratio = (exp_dA - 1.0) / neg_exp
            B_int_n = Value(ratio, (delta_A,), (
                (clamped_dA * exp_dA - exp_dA + 1.0) / (clamped_dA ** 2)
                if abs(clamped_dA) > 1e-8 else 0.5,
            )) * B[n]

        # Split the input weight between current and previous timestep.
        # This is the key structural difference from ZOH: the recurrence
        # now depends on x_{t-1}, creating an implicit size-2 convolution.
        B_bar_curr.append(B_int_n * alpha)
        B_bar_prev.append(B_int_n * (1.0 - alpha))

    return A_bar, B_bar_curr, B_bar_prev


# === SSM FORWARD PASS (PER DISCRETIZATION) ===

def ssm_forward(
    params: dict,
    x_seq: list[float],
    delta_seq: list[float],
    discretize_fn,
    task: str = "sine",
):
    """Run SSM forward pass with the given discretization method.

    Processes the sequence step-by-step, accumulating state h_t.
    For trapezoidal: also passes x_{t-1} to the discretization.

    The state h is N_STATE-dimensional. At each step:
    1. Discretize A, B using the current delta
    2. Update state: h_t = A_bar * h_{t-1} + B_bar * x_t
    3. Compute output: y_t = C^T h_t + D * x_t
    """
    log_A = params['log_A']
    B = params['B']
    C = params['C']
    D = params['D'][0]

    seq_len = len(x_seq)
    h = [Value(0.0) for _ in range(N_STATE)]
    outputs = []

    is_trapezoidal = (discretize_fn is trapezoidal_discretize)

    # For trapezoidal method, we need x_{t-1} — initialize to zero
    x_prev = Value(0.0)

    for t in range(seq_len):
        x_t = Value(float(x_seq[t]))
        delta_t = Value(float(delta_seq[t]))

        if is_trapezoidal:
            A_bar, B_bar_curr, B_bar_prev = trapezoidal_discretize(
                log_A, B, delta_t
            )

            # State update with both x_t and x_{t-1} — the implicit convolution.
            # This creates h_t = f(h_{t-1}, x_t, x_{t-1}), a size-2 conv window
            # that replaces the explicit short convolution in Mamba-1/2.
            h_new = []
            for n in range(N_STATE):
                h_n = A_bar[n] * h[n] + B_bar_curr[n] * x_t + B_bar_prev[n] * x_prev
                h_new.append(h_n)
        else:
            A_bar, B_bar = discretize_fn(log_A, B, delta_t)

            h_new = []
            for n in range(N_STATE):
                # Math: h_t[n] = A_bar[n] * h_{t-1}[n] + B_bar[n] * x_t
                h_n = A_bar[n] * h[n] + B_bar[n] * x_t
                h_new.append(h_n)

        h = h_new

        # Output: y_t = C^T h_t + D * x_t (feedthrough via D)
        y_t = D * x_t
        for n in range(N_STATE):
            y_t = y_t + C[n] * h[n]

        # For parity task, push through sigmoid to get probability
        if task == "parity":
            y_t = sigmoid_value(y_t)

        outputs.append(y_t)
        x_prev = x_t

    return outputs


# === TRAINING LOOP ===

def train_ssm(
    discretize_fn,
    method_name: str,
    train_data: list,
    test_data: list,
    task: str = "sine",
) -> dict:
    """Train a single SSM variant and return metrics.

    Each method gets identical architecture (N_STATE=4, same init distribution)
    and training setup (Adam, same LR schedule). The only variable is the
    discretization function — isolating its effect on learning.
    """
    params = init_ssm_params()
    param_list = collect_params(params)

    # Adam optimizer state: m (momentum) and v (variance) per parameter
    m = [0.0] * len(param_list)
    v = [0.0] * len(param_list)

    start = time.time()
    final_loss = 0.0

    print(f"\n--- Training {method_name} SSM ({task} task) ---")

    for step in range(NUM_STEPS):
        # Cycle through training data
        x_seq, delta_seq, target_seq = train_data[step % len(train_data)]

        # Forward pass
        outputs = ssm_forward(params, x_seq, delta_seq, discretize_fn, task)

        # Compute loss
        if task == "sine":
            loss = mse_loss(outputs, target_seq)
        else:
            loss = binary_cross_entropy_loss(outputs, target_seq)

        # Zero gradients before backward (accumulation bug otherwise)
        for p in param_list:
            p.grad = 0.0

        loss.backward()

        # Adam with linear LR decay
        lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)

        for i, p in enumerate(param_list):
            m[i] = BETA1 * m[i] + (1 - BETA1) * p.grad
            v[i] = BETA2 * v[i] + (1 - BETA2) * p.grad ** 2

            # Bias correction compensates for zero-initialization of m and v
            m_hat = m[i] / (1 - BETA1 ** (step + 1))
            v_hat = v[i] / (1 - BETA2 ** (step + 1))

            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)

        final_loss = loss.data

        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{NUM_STEPS:>4} | loss: {loss.data:.4f}")

    train_time = time.time() - start

    # === INFERENCE ===
    # Evaluate on test set to measure generalization
    test_metric = evaluate_ssm(params, test_data, discretize_fn, task)

    return {
        'method': method_name,
        'final_loss': final_loss,
        'test_metric': test_metric,
        'train_time': train_time,
        'params': params,
    }


def evaluate_ssm(
    params: dict,
    test_data: list,
    discretize_fn,
    task: str,
) -> float:
    """Evaluate trained SSM on test data.

    Returns MSE for sine task, accuracy for parity task.
    Uses .data (plain floats) — no autograd needed for inference.
    """
    if task == "sine":
        total_mse = 0.0
        for x_seq, delta_seq, target_seq in test_data:
            outputs = ssm_forward(params, x_seq, delta_seq, discretize_fn, task)
            mse = sum(
                (out.data - tgt) ** 2 for out, tgt in zip(outputs, target_seq)
            ) / len(target_seq)
            total_mse += mse
        return total_mse / len(test_data)
    else:
        correct = 0
        total = 0
        for x_seq, delta_seq, target_seq in test_data:
            outputs = ssm_forward(params, x_seq, delta_seq, discretize_fn, task)
            for out, tgt in zip(outputs, target_seq):
                pred = 1.0 if out.data > 0.5 else 0.0
                if pred == tgt:
                    correct += 1
                total += 1
        return correct / total


# === MAIN ===

def main():
    overall_start = time.time()

    # --- Generate synthetic data ---
    print("Generating synthetic data...")
    sine_train = generate_sine_data(NUM_TRAIN, SEQ_LEN)
    sine_test = generate_sine_data(NUM_TEST, SEQ_LEN)
    parity_train = generate_parity_data(NUM_TRAIN, SEQ_LEN)
    parity_test = generate_parity_data(NUM_TEST, SEQ_LEN)
    print(f"  Sine: {NUM_TRAIN} train, {NUM_TEST} test sequences (len={SEQ_LEN})")
    print(f"  Parity: {NUM_TRAIN} train, {NUM_TEST} test sequences (len={SEQ_LEN})")

    # --- Train all three methods on sine task ---
    print("\n" + "=" * 65)
    print("TASK 1: IRREGULAR SINE WAVE PREDICTION")
    print("=" * 65)

    euler_sine = train_ssm(
        euler_discretize, "Euler", sine_train, sine_test, "sine"
    )
    zoh_sine = train_ssm(
        zoh_discretize, "ZOH (Mamba-1/2)", sine_train, sine_test, "sine"
    )
    trap_sine = train_ssm(
        trapezoidal_discretize, "Trapezoidal (Mamba-3)",
        sine_train, sine_test, "sine"
    )

    # --- Train all three methods on parity task ---
    print("\n" + "=" * 65)
    print("TASK 2: PARITY PREFIX SUMS (RUNNING XOR)")
    print("=" * 65)

    euler_parity = train_ssm(
        euler_discretize, "Euler", parity_train, parity_test, "parity"
    )
    zoh_parity = train_ssm(
        zoh_discretize, "ZOH (Mamba-1/2)", parity_train, parity_test, "parity"
    )
    trap_parity = train_ssm(
        trapezoidal_discretize, "Trapezoidal (Mamba-3)",
        parity_train, parity_test, "parity"
    )

    # === COMPARISON TABLE ===

    # --- Table 1: Sine task results ---
    print("\n" + "=" * 65)
    print("COMPARISON: SINE TASK")
    print("=" * 65)

    hdr = f"{'Method':<25} {'Final Loss':>12} {'MSE (test)':>12} {'Time (s)':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in [euler_sine, zoh_sine, trap_sine]:
        print(
            f"{r['method']:<25} {r['final_loss']:>12.4f} "
            f"{r['test_metric']:>12.4f} {r['train_time']:>10.2f}"
        )

    # --- Table 2: Parity task results ---
    print("\n" + "=" * 65)
    print("COMPARISON: PARITY TASK")
    print("=" * 65)

    hdr = f"{'Method':<25} {'Final Loss':>12} {'Accuracy':>10} {'Time (s)':>10}"
    print(hdr)
    print("-" * len(hdr))
    for r in [euler_parity, zoh_parity, trap_parity]:
        print(
            f"{r['method']:<25} {r['final_loss']:>12.4f} "
            f"{r['test_metric']:>10.1%} {r['train_time']:>10.2f}"
        )

    # --- Table 3: Effective recurrence structure ---
    # This table is the conceptual takeaway: how the recurrence differs
    print("\n" + "=" * 65)
    print("EFFECTIVE RECURRENCE STRUCTURE")
    print("=" * 65)

    hdr = (
        f"{'Method':<16} {'A_bar formula':<22} "
        f"{'Depends on x_{t-1}?':<22} {'Implicit conv size':>18}"
    )
    print(hdr)
    print("-" * len(hdr))
    print(
        f"{'Euler':<16} {'I + Δ A':<22} "
        f"{'No':<22} {'0':>18}"
    )
    print(
        f"{'ZOH':<16} {'exp(Δ A)':<22} "
        f"{'No':<22} {'0':>18}"
    )
    print(
        f"{'Trapezoidal':<16} {'exp(Δ A)':<22} "
        f"{'Yes':<22} {'2':>18}"
    )

    # === STABILITY ANALYSIS ===

    # This is the "aha" section: Euler diverges at large delta, exponential
    # methods (ZOH and trapezoidal) remain bounded for any delta > 0.
    # The stability condition for Euler is |1 + delta * a_n| < 1.
    # For a_n < 0: delta < -2/a_n is the stability boundary.

    print("\n" + "=" * 65)
    print("STABILITY ANALYSIS: |A_bar| FOR VARYING DELTA")
    print("=" * 65)
    print("(Using a_n = -0.5, representative diagonal element)")
    print()

    a_n = -0.5  # representative eigenvalue (moderate decay)

    hdr = f"{'delta':>8} {'|Euler|':>10} {'|ZOH|':>10} {'|Trap|':>10} {'Euler stable?':>15}"
    print(hdr)
    print("-" * len(hdr))

    # Sweep delta from small (0.1) to large (25.0)
    delta_values = [0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 8.0, 10.0, 15.0, 20.0, 25.0]

    for delta in delta_values:
        # Euler: |1 + delta * a_n|
        euler_abs = abs(1.0 + delta * a_n)

        # ZOH and Trapezoidal share the same A_bar = exp(delta * a_n)
        # For a_n < 0 and delta > 0, this is always in (0, 1) — unconditionally stable
        zoh_abs = math.exp(delta * a_n)

        # Euler stability boundary: |1 + delta * a_n| < 1
        # For a_n = -0.5: stable when delta < 4.0
        stable = "YES" if euler_abs < 1.0 else "NO — DIVERGES"

        print(
            f"{delta:>8.1f} {euler_abs:>10.4f} {zoh_abs:>10.4f} "
            f"{zoh_abs:>10.4f} {stable:>15}"
        )

    # Intuition: the exponential map exp(delta * a_n) is a natural compression — it maps
    # the entire negative real line to (0, 1). Euler's linear approximation 1 + delta * a_n
    # overshoots past -1 when the step size is too large, causing the state to oscillate
    # and grow without bound. This is why production SSMs always use exp-based discretization.
    print()
    print("Key insight: Euler's linear approximation breaks at delta > -2/a_n = "
          f"{-2.0 / a_n:.1f}.")
    print("ZOH/Trapezoidal use exp(), which maps ANY delta to (0, 1) — always stable.")

    total_time = time.time() - overall_start
    print(f"\nTotal runtime: {total_time:.1f}s")


if __name__ == "__main__":
    main()
