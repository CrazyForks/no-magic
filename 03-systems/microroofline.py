"""
A state update running 100x more FLOPs can be faster on real hardware if it shifts from
memory-bound to compute-bound — the roofline model explains why.
"""
# Reference: Mamba-3 (arXiv:2603.15569), Section 4 — MIMO and Hardware Efficiency
# Williams et al. 2009 — Roofline: An Insightful Visual Performance Model
#
# === TRADEOFFS ===
# + Roofline model: universal framework for memory-bound vs. compute-bound reasoning
# + MIMO: increases arithmetic intensity by batching scalar ops into matmuls
# + Applicable beyond SSMs: Flash Attention, batched inference, kernel optimization
# - Pure Python timing is noisy — trends visible, absolute numbers not meaningful
# - CPU roofline differs from GPU roofline (effect is 100x more dramatic on GPU)
# WHEN TO USE: when optimizing any operation's hardware utilization
# WHEN NOT TO: when the bottleneck is algorithmic, not hardware

# ─── Connection to other no-magic scripts ───
# microssm.py         — SISO SSM implementation (memory-bound baseline)
# microflash.py       — restructuring attention for SRAM (same roofline principle)
# microspeculative.py — draft-verify tradeoff (hardware-aware algorithm design)
# microparallel.py    — tensor/pipeline parallelism (also roofline-aware)

from __future__ import annotations

import math
import random
import time

random.seed(42)


# === CONSTANTS AND HYPERPARAMETERS ===

# Roofline parameters (estimated for M-series CPU, single core)
# These are rough estimates — the point is relative positioning, not absolute accuracy
PEAK_FLOPS = 50e9          # ~50 GFLOPS single-core (M-series, FP64)
PEAK_BANDWIDTH = 100e9     # ~100 GB/s memory bandwidth (M-series unified memory)
RIDGE_POINT = PEAK_FLOPS / PEAK_BANDWIDTH  # ~0.5 FLOPs/byte

# SSM dimensions for SISO/MIMO comparison
N_STATE = 16               # state dimension
N_EMBD = 8                 # embedding dimension (SISO input width)
MIMO_RANKS = [1, 4, 8, 16]  # rank sweep for MIMO

# Sequence lengths for the SISO vs MIMO timing comparison
SEQ_LENS = [1000, 4000, 16000]

# Training (for Phase 4)
LEARNING_RATE = 0.01
NUM_STEPS = 400
BETA1 = 0.85
BETA2 = 0.99
EPS_ADAM = 1e-8             # matches PyTorch default; prevents div-by-zero in Adam
SEQ_LEN = 64
NUM_TRAIN = 200
NUM_TEST = 50

# Signpost: H100 has ~1000 TFLOPS (FP16) and ~3.35 TB/s bandwidth,
# giving a ridge point of ~300 FLOPs/byte. The memory-bound gap is
# 100x more dramatic on GPU — SISO wastes 99% of compute capability.


# === SCALAR AUTOGRAD ENGINE ===

# Only used in Phase 4 (training comparison). Phases 1-3 use plain floats
# for timing accuracy — autograd overhead would distort measurements.

class Value:
    """Scalar with reverse-mode automatic differentiation."""
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data: float, children: tuple = (), local_grads: tuple = ()) -> None:
        self.data = float(data)
        self.grad = 0.0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1.0, 1.0))

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        # d(a*b)/da = b, d(a*b)/db = a
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, exponent: float) -> Value:
        # d(x^n)/dx = n * x^(n-1)
        return Value(
            self.data ** exponent, (self,),
            (exponent * self.data ** (exponent - 1),)
        )

    def __neg__(self) -> Value:
        return self * -1

    def __radd__(self, other: float) -> Value:
        return self + other

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other)

    def __rsub__(self, other: float) -> Value:
        return other + (-self)

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        return self * (other ** -1)

    def __rtruediv__(self, other: float) -> Value:
        return other * (self ** -1)

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        return Value(t, (self,), (1 - t ** 2,))

    def exp(self) -> Value:
        e = math.exp(min(self.data, 80.0))  # clamp prevents overflow
        return Value(e, (self,), (e,))

    def log(self) -> Value:
        # Clamp to 1e-10 before log to prevent log(0) = -inf.
        # Gradient uses the clamped value to stay finite.
        clamped = max(self.data, 1e-10)
        return Value(math.log(clamped), (self,), (1.0 / clamped,))

    def relu(self) -> Value:
        return Value(max(0.0, self.data), (self,), (float(self.data > 0),))

    def backward(self) -> None:
        """Reverse-mode autodiff: topological sort then chain-rule propagation."""
        topo: list[Value] = []
        visited: set[int] = set()

        def build_topo(v: Value) -> None:
            vid = id(v)
            if vid not in visited:
                visited.add(vid)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1.0
        for v in reversed(topo):
            for child, lg in zip(v._children, v._local_grads):
                child.grad += lg * v.grad


# --- AUTOGRAD IN THIS SCRIPT ---
# This Value class follows the canonical interface exactly.
# Autograd is used only for the SISO vs. MIMO training comparison (Phase 4).
# The roofline measurement harness (Phases 1-3) uses plain floats for timing accuracy.
# See docs/autograd-interface.md for the full specification.


# === ROOFLINE MODEL ===

# The roofline model captures the fundamental tension in hardware:
# every operation is limited by EITHER memory bandwidth OR compute throughput,
# whichever is the tighter bottleneck. The crossover is the "ridge point."

def arithmetic_intensity(flops: float, bytes_transferred: float) -> float:
    """FLOPs per byte of memory traffic — the x-axis of the roofline plot.

    # Math: AI = FLOPs / Bytes
    # Units: FLOPs/byte (dimensionless ratio, but conventionally written this way)
    # The ridge point is where AI = peak_flops / peak_bandwidth.
    # Below the ridge: memory-bound (bandwidth ceiling limits throughput).
    # Above the ridge: compute-bound (FLOP ceiling limits throughput).
    """
    if bytes_transferred == 0:
        return float('inf')
    return flops / bytes_transferred


def theoretical_throughput(ai: float, peak_flops: float, peak_bandwidth: float) -> float:
    """Achievable throughput given arithmetic intensity.

    # Math: throughput = min(peak_flops, ai * peak_bandwidth)
    # This is the roofline equation — two regimes joined at the ridge point.
    # Left of ridge: throughput grows linearly with AI (memory-bound slope).
    # Right of ridge: throughput is flat at peak_flops (compute-bound ceiling).
    """
    return min(peak_flops, ai * peak_bandwidth)


# === MEASUREMENT HARNESS ===

# Pure Python list operations for timing. These are the operations we'll
# place on the roofline plot to show how arithmetic intensity determines
# which ceiling limits performance.

def measure_time(fn, warmup: int = 2, trials: int = 5) -> float:
    """Wall-clock timing with warmup runs discarded.

    # Signpost: on GPU you'd use CUDA events for nanosecond precision.
    # On CPU, time.perf_counter() gives microsecond resolution — enough
    # to see trends, but individual measurements are noisy.
    """
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(trials):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return min(times)  # min reduces noise from OS scheduling jitter


# --- FLOP and byte counting ---

def count_flops_vecadd(n: int) -> int:
    """Vector addition: n additions = n FLOPs."""
    return n


def count_bytes_vecadd(n: int) -> int:
    """Vector addition: read 2 vectors + write 1 = 3n * 8 bytes (float64)."""
    return 3 * n * 8


def count_flops_outer(n: int, m: int) -> int:
    """Outer product: 2nm FLOPs (n*m multiplies + n*m additions for accumulation).

    # In the SSM context, the outer product is B*x^T which then gets ADDED to A*h.
    # That addition is part of the operation, so we count 2nm.
    """
    return 2 * n * m


def count_bytes_outer(n: int, m: int) -> int:
    """Outer product bytes: read inputs (n + m floats), output stays in registers.

    # On GPU, the roofline counts HBM traffic. The outer product reads two vectors
    # from HBM; the n*m result is consumed immediately by the state accumulation
    # (fused in the same kernel). Only input reads count for the roofline.
    # This is why SISO has AI ~2: 2nm FLOPs / (n+m)*8 bytes.
    """
    return (n + m) * 8


def count_flops_matmul(n: int, m: int, k: int) -> int:
    """Matrix multiply [n x k] @ [k x m]: 2nmk FLOPs (n*m dot products, each 2k ops)."""
    return 2 * n * m * k


def count_bytes_matmul(n: int, m: int, k: int) -> int:
    """Matmul bytes: read A[n,k] + B[k,m], output stays in registers.

    # Same principle as outer product: on GPU the output matrix C is accumulated
    # in registers/SRAM before being written back. The roofline bottleneck is
    # reading the input matrices from HBM. This gives matmul its high AI.
    """
    return (n * k + k * m) * 8


# --- Pure Python matrix operations ---

def vec_add(a: list[float], b: list[float]) -> list[float]:
    """Element-wise vector addition."""
    return [a[i] + b[i] for i in range(len(a))]


def outer_product(a: list[float], b: list[float]) -> list[list[float]]:
    """Outer product: result[i][j] = a[i] * b[j]."""
    return [[a[i] * b[j] for j in range(len(b))] for i in range(len(a))]


def matmul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    """Matrix multiply A[n,k] @ B[k,m] -> C[n,m]."""
    n = len(a)
    k = len(a[0])
    m = len(b[0])
    result = [[0.0] * m for _ in range(n)]
    for i in range(n):
        for j in range(m):
            s = 0.0
            for p in range(k):
                s += a[i][p] * b[p][j]
            result[i][j] = s
    return result


# === FOUR OPERATIONS ON THE ROOFLINE ===

# We measure four operations that span the roofline from deep memory-bound
# (vector add) to compute-bound (large matmul). The SISO-to-MIMO transition
# in SSMs follows this exact trajectory.

def run_roofline_operations() -> list[tuple[str, float, float, float]]:
    """Measure four operations and compute their roofline position.

    Returns list of (name, arithmetic_intensity, measured_time, flops).

    # We use larger dimensions than the SSM training section to make the
    # AI values clear and to generate measurable timing differences.
    # The AI formula depends on dimension ratios, not absolute sizes.
    """
    results = []

    # Dimensions chosen so AI matches the spec's target values.
    # For outer product a[n] * b[m]: AI = 2nm / (n+m)*8
    # For matmul A[n,k] @ B[k,m]: AI = 2nmk / (nk+km)*8
    # Using n=m=64 makes the math clean and timing measurable.
    dim = 64

    # (a) Vector addition: AI ≈ 0.33 FLOPs/byte
    # Reading 2 vectors, writing 1, doing 1 FLOP per element.
    # This is the most memory-bound operation possible — almost no computation
    # per byte moved.
    n_vec = 4096
    a_vec = [random.random() for _ in range(n_vec)]
    b_vec = [random.random() for _ in range(n_vec)]
    flops_va = count_flops_vecadd(n_vec)
    bytes_va = count_bytes_vecadd(n_vec)
    ai_va = arithmetic_intensity(flops_va, bytes_va)
    t_va = measure_time(lambda: vec_add(a_vec, b_vec))
    results.append(("Vec Add", ai_va, t_va, flops_va))

    # (b) Outer product (SISO state update): AI ≈ 2
    # B[n] * x[m]^T where both vectors are length 64.
    # AI = 2*64*64 / (64+64)*8 = 8192/1024 = 8. With smaller dims (n=16, m=16):
    # AI = 2*16*16 / (16+16)*8 = 512/256 = 2.0. Use n=m=16 for AI ~2.
    n_op, m_op = 16, 16
    a_outer = [random.random() for _ in range(n_op)]
    b_outer = [random.random() for _ in range(m_op)]
    flops_op = count_flops_outer(n_op, m_op)
    bytes_op = count_bytes_outer(n_op, m_op)
    ai_op = arithmetic_intensity(flops_op, bytes_op)
    t_op = measure_time(lambda: outer_product(a_outer, b_outer))
    results.append(("Outer (SISO)", ai_op, t_op, flops_op))

    # (c) Matrix multiply rank-4 (MIMO): AI ≈ 8
    # A[64,4] @ B[4,64]: AI = 2*64*64*4 / (64*4 + 4*64)*8 = 32768/4096 = 8.0
    # The same total output dimensions (64x64) but computed as a matmul with
    # inner dimension 4 (rank-4 update). Reusing A's rows across B's columns
    # is what gives matmul its AI advantage.
    rank4 = 4
    mat_a4 = [[random.random() for _ in range(rank4)] for _ in range(dim)]
    mat_b4 = [[random.random() for _ in range(dim)] for _ in range(rank4)]
    flops_mm4 = count_flops_matmul(dim, dim, rank4)
    bytes_mm4 = count_bytes_matmul(dim, dim, rank4)
    ai_mm4 = arithmetic_intensity(flops_mm4, bytes_mm4)
    t_mm4 = measure_time(lambda: matmul(mat_a4, mat_b4))
    results.append(("Matmul r=4 (MIMO)", ai_mm4, t_mm4, flops_mm4))

    # (d) Matrix multiply rank-16 (MIMO): AI ≈ 32
    # A[64,16] @ B[16,64]: AI = 2*64*64*16 / (64*16 + 16*64)*8 = 131072/16384 = 8.
    # To get AI ~32, we need larger n,m relative to k. Use n=m=128, k=16:
    # AI = 2*128*128*16 / (128*16 + 16*128)*8 = 524288/32768 = 16. Still not 32.
    # Use n=m=256, k=16: AI = 2*256*256*16 / (256*16+16*256)*8 = 2097152/65536 = 32.
    big = 256
    rank16 = 16
    mat_a16 = [[random.random() for _ in range(rank16)] for _ in range(big)]
    mat_b16 = [[random.random() for _ in range(big)] for _ in range(rank16)]
    flops_mm16 = count_flops_matmul(big, big, rank16)
    bytes_mm16 = count_bytes_matmul(big, big, rank16)
    ai_mm16 = arithmetic_intensity(flops_mm16, bytes_mm16)
    t_mm16 = measure_time(lambda: matmul(mat_a16, mat_b16))
    results.append(("Matmul r=16 (MIMO)", ai_mm16, t_mm16, flops_mm16))

    return results


# === SSM STATE UPDATE: SISO VS. MIMO ===

# The core insight from Mamba-3 Section 4: SISO state updates are memory-bound
# because the outer product B * x_t has low arithmetic intensity. MIMO batches
# multiple inputs into a matmul, shifting the operation rightward on the roofline.

def siso_state_update(h: list[list[float]], A: list[float], B: list[float],
                      x: list[float]) -> list[list[float]]:
    """SISO: h_t = diag(A) * h_{t-1} + outer(B, x_t)

    # Math: H[n, d] = A[n] * H[n, d] + B[n] * x[d]
    # The outer product B * x_t has arithmetic intensity ~2 FLOPs/byte.
    # On H100: 2/300 = 0.7% compute utilization. The GPU is 99% idle.
    """
    n = len(h)
    d = len(h[0])
    result = [[0.0] * d for _ in range(n)]
    for i in range(n):
        for j in range(d):
            result[i][j] = A[i] * h[i][j] + B[i] * x[j]
    return result


def mimo_state_update(H: list[list[float]], A: list[float],
                      B: list[list[float]], X: list[list[float]]) -> list[list[float]]:
    """MIMO: H_t = diag(A) * H_{t-1} + B @ X_t^T

    # Math: H[n, d] = A[n] * H[n, d] + (B[n, r] @ X[r, d])
    # Arithmetic intensity: ~2r FLOPs/byte for B @ X_t^T
    # At rank=16: 32 FLOPs/byte, approaching the ridge point.
    # More FLOPs total, but higher utilization → faster wall-clock time.
    """
    n = len(H)
    d = len(H[0])
    r = len(B[0])
    result = [[0.0] * d for _ in range(n)]
    for i in range(n):
        for j in range(d):
            # Decay existing state
            val = A[i] * H[i][j]
            # Matmul contribution: sum_k B[i][k] * X[k][j]
            for k in range(r):
                val += B[i][k] * X[k][j]
            result[i][j] = val
    return result


def run_ssm_comparison() -> list[tuple[str, int, float, float, float]]:
    """Compare SISO vs MIMO state updates across sequence lengths.

    Returns list of (name, seq_len, measured_time, total_flops, ai).
    """
    results = []

    # Initialize shared state and parameters
    A = [0.9 + 0.1 * random.random() for _ in range(N_STATE)]  # decay factors < 1

    for seq_len in SEQ_LENS:
        # --- SISO ---
        B_siso = [random.gauss(0, 0.1) for _ in range(N_STATE)]
        h_siso = [[0.0] * N_EMBD for _ in range(N_STATE)]
        x_seq = [[random.gauss(0, 1) for _ in range(N_EMBD)] for _ in range(seq_len)]

        def run_siso():
            h = [[0.0] * N_EMBD for _ in range(N_STATE)]
            for t in range(seq_len):
                h = siso_state_update(h, A, B_siso, x_seq[t])
            return h

        t_siso = measure_time(run_siso, warmup=1, trials=3)
        # Per step: N_STATE*N_EMBD for A*h decay + 2*N_STATE*N_EMBD for outer+add
        flops_per_step = 3 * N_STATE * N_EMBD
        total_flops_siso = flops_per_step * seq_len
        # Bytes: read h[N_STATE, N_EMBD] + B[N_STATE] + x[N_EMBD] from memory
        bytes_per_step = (N_STATE * N_EMBD + N_STATE + N_EMBD) * 8
        ai_siso = arithmetic_intensity(flops_per_step, bytes_per_step)
        results.append(("SISO", seq_len, t_siso, total_flops_siso, ai_siso))

        # --- MIMO rank-16 ---
        rank = 16
        B_mimo = [[random.gauss(0, 0.1) for _ in range(rank)]
                  for _ in range(N_STATE)]
        X_seq = [[[random.gauss(0, 1) for _ in range(N_EMBD)]
                  for _ in range(rank)] for _ in range(seq_len)]

        def run_mimo():
            h = [[0.0] * N_EMBD for _ in range(N_STATE)]
            for t in range(seq_len):
                h = mimo_state_update(h, A, B_mimo, X_seq[t])
            return h

        t_mimo = measure_time(run_mimo, warmup=1, trials=3)
        # Per step: N_STATE*N_EMBD for A*h + 2*N_STATE*N_EMBD*rank for matmul
        flops_per_step_mimo = N_STATE * N_EMBD + 2 * N_STATE * N_EMBD * rank
        total_flops_mimo = flops_per_step_mimo * seq_len
        # Bytes: read h[N_STATE, N_EMBD] + B[N_STATE, rank] + X[rank, N_EMBD]
        bytes_per_step_mimo = (N_STATE * N_EMBD + N_STATE * rank + rank * N_EMBD) * 8
        ai_mimo = arithmetic_intensity(flops_per_step_mimo, bytes_per_step_mimo)
        results.append(("MIMO-16", seq_len, t_mimo, total_flops_mimo, ai_mimo))

    return results


# === DATA GENERATION ===

# Synthetic next-value prediction task for the training comparison.
# Each sequence is a sum of two sine waves at different frequencies —
# SSMs should capture the temporal structure through their state.

def generate_sequences(n_sequences: int, seq_len: int) -> list[list[float]]:
    """Generate sequences of sin(f1*t) + sin(f2*t) with random frequencies.

    Two frequencies force the model to use multiple state dimensions to
    track the signal — a single-frequency sine could be captured by one state.
    """
    sequences = []
    for _ in range(n_sequences):
        f1 = random.uniform(0.05, 0.3)
        f2 = random.uniform(0.1, 0.5)
        phase = random.uniform(0, 2 * math.pi)
        seq = [math.sin(f1 * t + phase) + math.sin(f2 * t)
               for t in range(seq_len + 1)]
        sequences.append(seq)
    return sequences


# === TRAINING COMPARISON: SISO VS. MIMO ===

# Phase 4: train matched-parameter SISO and MIMO SSMs on the same task.
# This demonstrates that MIMO isn't just faster on hardware — the matmul
# formulation can also improve model quality by processing richer input.

def softmax(logits: list[Value]) -> list[Value]:
    """Numerically stable softmax: subtract max before exp to prevent overflow."""
    max_val = max(v.data for v in logits)
    exp_vals = [(v - max_val).exp() for v in logits]
    total = sum(exp_vals)
    return [e / total for e in exp_vals]


def train_siso_ssm(train_data: list[list[float]],
                   test_data: list[list[float]]) -> tuple[float, float, float]:
    """Train SISO SSM on next-value prediction. Returns (loss, test_mse, time)."""
    t0 = time.time()

    # Parameters: A (diagonal decay), B (input projection), C (output projection)
    # SISO: scalar input per step, so B and C are vectors
    n_state = 4  # small for tractable autograd
    log_A = [Value(random.gauss(-1.0, 0.3)) for _ in range(n_state)]
    B = [Value(random.gauss(0, 0.1)) for _ in range(n_state)]
    C = [Value(random.gauss(0, 0.1)) for _ in range(n_state)]

    params = log_A + B + C
    # Signpost: 12 parameters total. Production SSMs have millions.
    # We use this tiny size so training completes in seconds with scalar autograd.

    # Adam optimizer state
    m_adam = [0.0] * len(params)
    v_adam = [0.0] * len(params)

    print("\n  Training SISO SSM...")
    final_loss = 0.0
    for step in range(NUM_STEPS):
        seq = train_data[step % len(train_data)]

        # Forward: process sequence step by step
        # Math: h_t = diag(exp(log_A)) * h_{t-1} + B * x_t
        #        y_t = C^T h_t
        h = [Value(0.0) for _ in range(n_state)]
        loss = Value(0.0)

        for t in range(len(seq) - 1):
            x_t = Value(seq[t])
            target = seq[t + 1]

            # State update with exp(log_A) parameterization for stability
            new_h = []
            for n in range(n_state):
                # A_n = exp(log_A_n) guarantees 0 < A_n, and we init log_A < 0
                # so A_n < 1 (stable decay)
                a_n = log_A[n].exp()
                new_h.append(a_n * h[n] + B[n] * x_t)
            h = new_h

            # Output: y_t = C^T h_t
            y_t = sum(C[n] * h[n] for n in range(n_state))

            # MSE loss per step
            diff = y_t - target
            loss = loss + diff * diff

        loss = loss * (1.0 / (len(seq) - 1))

        # Backward + Adam update
        for p in params:
            p.grad = 0.0
        loss.backward()

        lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)
        for i, p in enumerate(params):
            m_adam[i] = BETA1 * m_adam[i] + (1 - BETA1) * p.grad
            v_adam[i] = BETA2 * v_adam[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_adam[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_adam[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)

        final_loss = loss.data
        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{NUM_STEPS:>4} | loss: {loss.data:.4f}")

    # Evaluate on test data
    test_mse = 0.0
    for seq in test_data:
        h = [Value(0.0) for _ in range(n_state)]
        seq_err = 0.0
        for t in range(len(seq) - 1):
            x_t = Value(seq[t])
            for n in range(n_state):
                a_n = log_A[n].exp()
                h[n] = a_n * h[n] + B[n] * x_t
            y_t = sum(C[n].data * h[n].data for n in range(n_state))
            seq_err += (y_t - seq[t + 1]) ** 2
        test_mse += seq_err / (len(seq) - 1)
    test_mse /= len(test_data)

    elapsed = time.time() - t0
    return final_loss, test_mse, elapsed


def train_mimo_ssm(train_data: list[list[float]],
                   test_data: list[list[float]],
                   rank: int = 4) -> tuple[float, float, float]:
    """Train MIMO SSM on next-value prediction. Returns (loss, test_mse, time).

    # MIMO processes `rank` consecutive inputs at once via matmul.
    # More parameters per step, but the matmul structure is hardware-friendly.
    # We match total parameter count to SISO by adjusting n_state.
    """
    t0 = time.time()

    # Matched parameters: SISO has n_state*(1 + 1 + 1) = 3*n_state = 12 params
    # MIMO has n_state*(1 + rank + rank) = n_state*(1 + 2*rank)
    # For rank=4, n_state=1 gives 9 params. Use n_state=2 for 18 (close enough).
    n_state = 2
    log_A = [Value(random.gauss(-1.0, 0.3)) for _ in range(n_state)]
    # B: [n_state, rank] — projects rank-dimensional input to state
    B = [[Value(random.gauss(0, 0.1)) for _ in range(rank)]
         for _ in range(n_state)]
    # C: [n_state, rank] — projects state to rank-dimensional output
    C = [[Value(random.gauss(0, 0.1)) for _ in range(rank)]
         for _ in range(n_state)]

    params = log_A[:]
    for row in B:
        params.extend(row)
    for row in C:
        params.extend(row)

    m_adam = [0.0] * len(params)
    v_adam = [0.0] * len(params)

    print(f"\n  Training MIMO (rank={rank}) SSM...")
    final_loss = 0.0
    for step in range(NUM_STEPS):
        seq = train_data[step % len(train_data)]

        h = [Value(0.0) for _ in range(n_state)]
        loss = Value(0.0)
        n_predictions = 0

        # Process sequence in chunks of `rank` tokens
        for t in range(0, len(seq) - rank, rank):
            x_chunk = [Value(seq[t + r]) for r in range(rank)]
            targets = [seq[t + r + 1] for r in range(rank)]

            # State update: h_t = A * h_{t-1} + B @ x_chunk
            # Math: new_h[n] = exp(log_A[n]) * h[n] + sum_r B[n][r] * x_chunk[r]
            new_h = []
            for n in range(n_state):
                a_n = log_A[n].exp()
                val = a_n * h[n]
                for r in range(rank):
                    val = val + B[n][r] * x_chunk[r]
                new_h.append(val)
            h = new_h

            # Output: y_chunk[r] = sum_n C[n][r] * h[n]
            for r in range(rank):
                y_r = sum(C[n][r] * h[n] for n in range(n_state))
                diff = y_r - targets[r]
                loss = loss + diff * diff
                n_predictions += 1

        if n_predictions > 0:
            loss = loss * (1.0 / n_predictions)

        for p in params:
            p.grad = 0.0
        loss.backward()

        lr_t = LEARNING_RATE * (1 - step / NUM_STEPS)
        for i, p in enumerate(params):
            m_adam[i] = BETA1 * m_adam[i] + (1 - BETA1) * p.grad
            v_adam[i] = BETA2 * v_adam[i] + (1 - BETA2) * p.grad ** 2
            m_hat = m_adam[i] / (1 - BETA1 ** (step + 1))
            v_hat = v_adam[i] / (1 - BETA2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat ** 0.5 + EPS_ADAM)

        final_loss = loss.data
        if (step + 1) % 100 == 0 or step == 0:
            print(f"  step {step + 1:>4}/{NUM_STEPS:>4} | loss: {loss.data:.4f}")

    # Evaluate on test data
    test_mse = 0.0
    for seq in test_data:
        h = [0.0] * n_state
        seq_err = 0.0
        n_pred = 0
        for t in range(0, len(seq) - rank, rank):
            x_chunk = [seq[t + r] for r in range(rank)]
            new_h = []
            for n in range(n_state):
                a_n = math.exp(log_A[n].data)
                val = a_n * h[n]
                for r in range(rank):
                    val += B[n][r].data * x_chunk[r]
                new_h.append(val)
            h = new_h
            for r in range(rank):
                y_r = sum(C[n][r].data * h[n] for n in range(n_state))
                seq_err += (y_r - seq[t + r + 1]) ** 2
                n_pred += 1
        if n_pred > 0:
            test_mse += seq_err / n_pred
    test_mse /= len(test_data)

    elapsed = time.time() - t0
    return final_loss, test_mse, elapsed


# === ASCII ROOFLINE PLOT ===

# The signature output: a 60x20 character grid showing log-log axes with
# the roofline ceiling and measured operation positions.

def print_roofline(operations: list[tuple[str, float, float]],
                   peak_flops: float, peak_bandwidth: float) -> None:
    """Print ASCII roofline plot with measured operation positions.

    operations: list of (name, arithmetic_intensity, measured_gflops)

    Grid: 60 columns x 20 rows
    X-axis: log2(arithmetic intensity), range [0.01, 1000]
    Y-axis: log2(throughput in GFLOPS), range [0.001, peak in GFLOPS]
    Characters:
      . = grid point
      # = roofline ceiling (bandwidth slope + compute flat)
      A-Z = operation markers (labeled in legend below)
    """
    width = 60
    height = 20

    # Log-scale ranges
    x_min_log = math.log2(0.01)    # AI = 0.01
    x_max_log = math.log2(1000.0)  # AI = 1000
    y_min_log = math.log2(0.001)   # 0.001 GFLOPS
    peak_gflops = peak_flops / 1e9
    y_max_log = math.log2(peak_gflops * 2)  # headroom above peak

    def x_to_col(log_ai: float) -> int:
        frac = (log_ai - x_min_log) / (x_max_log - x_min_log)
        return max(0, min(width - 1, int(frac * (width - 1))))

    def y_to_row(log_tp: float) -> int:
        # Row 0 = top (high throughput), row height-1 = bottom (low throughput)
        frac = (log_tp - y_min_log) / (y_max_log - y_min_log)
        return max(0, min(height - 1, height - 1 - int(frac * (height - 1))))

    # Initialize grid with dots
    grid = [['·'] * width for _ in range(height)]

    # Draw roofline ceiling
    # Left of ridge: throughput = AI * bandwidth (memory-bound slope)
    # Right of ridge: throughput = peak_flops (compute-bound flat)
    ridge_ai = peak_flops / peak_bandwidth
    for col in range(width):
        frac = col / (width - 1)
        log_ai = x_min_log + frac * (x_max_log - x_min_log)
        ai = 2 ** log_ai
        tp = min(peak_gflops, ai * peak_bandwidth / 1e9)
        if tp > 0:
            log_tp = math.log2(max(tp, 1e-12))
            row = y_to_row(log_tp)
            if 0 <= row < height:
                grid[row][col] = '█'

    # Place operation markers
    markers = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    legend = []
    for idx, (name, ai, gflops) in enumerate(operations):
        marker = markers[idx % len(markers)]
        if ai > 0 and gflops > 0:
            log_ai = math.log2(max(ai, 1e-12))
            log_tp = math.log2(max(gflops, 1e-12))
            col = x_to_col(log_ai)
            row = y_to_row(log_tp)
            if 0 <= row < height and 0 <= col < width:
                grid[row][col] = marker
        legend.append((marker, name, ai, gflops))

    # Print the plot
    print("\n  Roofline Plot (log-log scale)")
    print("  " + "─" * (width + 2))

    # Y-axis labels at a few rows
    y_label_rows = {0, height // 4, height // 2, 3 * height // 4, height - 1}
    for row in range(height):
        frac = 1.0 - row / (height - 1)
        log_val = y_min_log + frac * (y_max_log - y_min_log)
        val = 2 ** log_val
        if row in y_label_rows:
            if val >= 1.0:
                label = f"{val:>6.1f}"
            else:
                label = f"{val:>6.3f}"
        else:
            label = "      "
        print(f"  {label} |{''.join(grid[row])}|")

    print("  " + " " * 6 + " " + "─" * (width + 2))

    # X-axis labels
    x_positions = [0, width // 4, width // 2, 3 * width // 4, width - 1]
    x_labels = "  " + " " * 7
    for i, pos in enumerate(x_positions):
        frac = pos / (width - 1)
        log_val = x_min_log + frac * (x_max_log - x_min_log)
        val = 2 ** log_val
        if val < 0.1:
            lbl = f"{val:.2f}"
        elif val < 10:
            lbl = f"{val:.1f}"
        else:
            lbl = f"{val:.0f}"
        # Approximate horizontal positioning
        if i == 0:
            x_labels += lbl
        else:
            gap = (pos - x_positions[i - 1]) * 1 - len(lbl)
            x_labels += " " * max(1, gap) + lbl
    print(x_labels)
    print("  " + " " * 7 + " " * (width // 2 - 10) + "Arithmetic Intensity (FLOPs/byte)")

    # Ridge point annotation
    print(f"\n  Ridge point: {ridge_ai:.2f} FLOPs/byte "
          f"(peak={peak_gflops:.0f} GFLOPS / {peak_bandwidth/1e9:.0f} GB/s)")

    # Legend
    print("\n  Legend:")
    hdr = f"  {'Mark':<6} {'Operation':<22} {'AI (F/B)':>10} {'GFLOPS':>10}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for marker, name, ai, gflops in legend:
        print(f"  {marker:<6} {name:<22} {ai:>10.2f} {gflops:>10.4f}")


# === WHAT THIS LOOKS LIKE ON GPU ===
#
# The roofline effect is 100x more dramatic on accelerators:
#
#                    CPU (M-series)     H100 GPU
#   Peak FLOPs       50 GFLOPS          1000 TFLOPS (FP16)
#   Peak Bandwidth   100 GB/s           3.35 TB/s
#   Ridge Point      0.5 FLOPs/byte     ~300 FLOPs/byte
#
#   SISO AI (~2):    40% utilization     0.7% utilization
#   MIMO-16 AI (~32): saturated          11% utilization
#   MIMO-128 AI:     N/A                 ~43% utilization
#
# On CPU, the gap between SISO and MIMO is visible but modest (Python overhead
# dominates). On GPU, SISO literally leaves 99.3% of the chip idle.
# This is why Mamba-3's MIMO formulation matters: it's not about model quality
# (though that improves too) — it's about making the hardware actually work.


# === MAIN ===

def main() -> None:
    total_start = time.time()

    print("=" * 70)
    print("  ROOFLINE MODEL: Memory-Bound vs. Compute-Bound Operations")
    print("  SISO vs. MIMO SSM State Updates as Case Study")
    print("=" * 70)

    # --- Phase 1: Roofline Fundamentals ---
    print(f"\n{'=' * 70}")
    print("  PHASE 1: ROOFLINE FUNDAMENTALS")
    print(f"{'=' * 70}")
    print(f"\n  Peak compute:    {PEAK_FLOPS/1e9:.0f} GFLOPS")
    print(f"  Peak bandwidth:  {PEAK_BANDWIDTH/1e9:.0f} GB/s")
    print(f"  Ridge point:     {RIDGE_POINT:.2f} FLOPs/byte")
    print(f"\n  Below the ridge ({RIDGE_POINT:.2f}): memory-bound — "
          f"bandwidth limits throughput")
    print(f"  Above the ridge ({RIDGE_POINT:.2f}): compute-bound — "
          f"FLOPs limit throughput")

    # --- Phase 2: Four operations on the roofline ---
    print(f"\n{'=' * 70}")
    print("  PHASE 2: FOUR OPERATIONS ON THE ROOFLINE")
    print(f"{'=' * 70}")

    ops = run_roofline_operations()

    hdr = (f"{'Operation':<22} {'FLOPs':>10} {'Bytes':>10} "
           f"{'AI (F/B)':>10} {'Time (us)':>12} {'GFLOPS':>10}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    roofline_points = []
    for name, ai, t, flops in ops:
        gflops = (flops / t) / 1e9 if t > 0 else 0
        tp_theoretical = theoretical_throughput(ai, PEAK_FLOPS, PEAK_BANDWIDTH) / 1e9
        # Estimate bytes from AI
        bytes_est = flops / ai if ai > 0 else 1
        print(f"{name:<22} {flops:>10} {int(bytes_est):>10} "
              f"{ai:>10.2f} {t*1e6:>12.1f} {gflops:>10.4f}")
        roofline_points.append((name, ai, gflops))

    # Intuition: as arithmetic intensity increases, the operation moves from
    # memory-bound (left of ridge) toward compute-bound (right of ridge).
    # Vector add lives deep in memory-bound territory; rank-16 matmul
    # approaches or crosses the ridge.
    print(f"\n  Observation: AI increases from {ops[0][1]:.2f} (vec add) to "
          f"{ops[-1][1]:.2f} (rank-16 matmul)")
    print(f"  This is the SISO→MIMO transition: same state update, "
          f"higher arithmetic intensity")

    # --- Phase 3: SSM State Update Comparison ---
    print(f"\n{'=' * 70}")
    print("  PHASE 3: SSM STATE UPDATE — SISO VS. MIMO")
    print(f"{'=' * 70}")
    print(f"\n  N_STATE={N_STATE}, N_EMBD={N_EMBD}, MIMO rank=16")

    ssm_results = run_ssm_comparison()

    hdr2 = (f"{'Method':<12} {'Seq Len':>8} {'Time (ms)':>12} "
            f"{'Total MFLOPS':>14} {'AI (F/B)':>10} {'us/step':>10}")
    print(f"\n{hdr2}")
    print("-" * len(hdr2))

    for name, seq_len, t, total_flops, ai in ssm_results:
        us_per_step = (t / seq_len) * 1e6
        mflops = total_flops / 1e6
        print(f"{name:<12} {seq_len:>8} {t*1000:>12.2f} "
              f"{mflops:>14.1f} {ai:>10.2f} {us_per_step:>10.2f}")

    # Print the SISO vs MIMO speedup for each sequence length
    print("\n  Throughput comparison (FLOPs/second):")
    hdr3 = f"{'Seq Len':>8} {'SISO MFLOPS/s':>15} {'MIMO MFLOPS/s':>15} {'MIMO/SISO FLOPs':>16}"
    print(f"  {hdr3}")
    print(f"  {'-' * len(hdr3)}")
    for i in range(0, len(ssm_results), 2):
        siso = ssm_results[i]
        mimo = ssm_results[i + 1]
        siso_mflops_s = (siso[3] / siso[2]) / 1e6 if siso[2] > 0 else 0
        mimo_mflops_s = (mimo[3] / mimo[2]) / 1e6 if mimo[2] > 0 else 0
        flop_ratio = mimo[3] / siso[3] if siso[3] > 0 else 0
        print(f"  {siso[1]:>8} {siso_mflops_s:>15.1f} {mimo_mflops_s:>15.1f} "
              f"{flop_ratio:>15.1f}x")

    # Signpost: in pure Python, per-operation interpreter overhead dominates.
    # MIMO does ~11x more FLOPs but achieves higher throughput (MFLOPS/s),
    # showing better computational efficiency. On GPU (where the overhead
    # per FLOP is nanoseconds, not microseconds), MIMO's higher AI translates
    # directly to faster wall-clock time despite more total FLOPs.
    print("\n  Note: MIMO does more total FLOPs but achieves higher throughput")
    print("  (MFLOPS/s), demonstrating better hardware utilization.")
    print("  On GPU, this efficiency gap translates to real wall-clock speedup.")

    # --- Phase 4: Training Comparison ---
    print(f"\n{'=' * 70}")
    print("  PHASE 4: TRAINING COMPARISON — SISO VS. MIMO SSM")
    print(f"{'=' * 70}")
    print(f"\n  Task: next-value prediction on dual-sine sequences")
    print(f"  Sequence length: {SEQ_LEN}, Training steps: {NUM_STEPS}")

    train_data = generate_sequences(NUM_TRAIN, SEQ_LEN)
    test_data = generate_sequences(NUM_TEST, SEQ_LEN)

    siso_loss, siso_mse, siso_time = train_siso_ssm(train_data, test_data)
    mimo_loss, mimo_mse, mimo_time = train_mimo_ssm(train_data, test_data, rank=4)

    # --- Training comparison table ---
    print(f"\n  Training Results:")
    thdr = f"{'Method':<25} {'Final Loss':>12} {'Test MSE':>12} {'Time (s)':>10}"
    print(f"  {thdr}")
    print(f"  {'-' * len(thdr)}")
    print(f"  {'SISO (scalar input)':<25} {siso_loss:>12.4f} {siso_mse:>12.4f} "
          f"{siso_time:>10.2f}")
    print(f"  {'MIMO (rank=4 input)':<25} {mimo_loss:>12.4f} {mimo_mse:>12.4f} "
          f"{mimo_time:>10.2f}")

    # --- Phase 5: ASCII Roofline Plot ---
    print(f"\n{'=' * 70}")
    print("  PHASE 5: ASCII ROOFLINE PLOT")
    print(f"{'=' * 70}")

    # Collect all measured points for the plot
    all_points = roofline_points[:]

    # Add SISO and MIMO from Phase 3 (use the longest sequence for clearest signal)
    for name, seq_len, t, total_flops, ai in ssm_results:
        if seq_len == SEQ_LENS[-1]:
            gflops = (total_flops / t) / 1e9 if t > 0 else 0
            all_points.append((f"SSM {name} @{seq_len//1000}K", ai, gflops))

    print_roofline(all_points, PEAK_FLOPS, PEAK_BANDWIDTH)

    # === ROOFLINE INTERPRETATION TABLE ===
    print(f"\n{'=' * 70}")
    print("  ROOFLINE INTERPRETATION")
    print(f"{'=' * 70}")

    ihdr = (f"{'Operation':<22} {'AI':>8} {'Regime':<16} "
            f"{'Utilization':>12}")
    print(f"\n{ihdr}")
    print("-" * len(ihdr))

    for name, ai, gflops in all_points:
        regime = "MEMORY-BOUND" if ai < RIDGE_POINT else "COMPUTE-BOUND"
        # Utilization: ratio of achieved to theoretical throughput
        tp_theory = theoretical_throughput(ai, PEAK_FLOPS, PEAK_BANDWIDTH) / 1e9
        util = (gflops / tp_theory * 100) if tp_theory > 0 else 0
        print(f"{name:<22} {ai:>8.2f} {regime:<16} {util:>11.1f}%")

    print(f"\n  Ridge point = {RIDGE_POINT:.2f} FLOPs/byte")
    print(f"  Operations below {RIDGE_POINT:.2f} are starved for data — "
          f"compute units sit idle.")
    print(f"  MIMO shifts the operation rightward, using more of the hardware.\n")

    # === SUMMARY ===
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print("""
  The roofline model has two regimes:
    1. MEMORY-BOUND (AI < ridge): throughput = AI * bandwidth
    2. COMPUTE-BOUND (AI > ridge): throughput = peak FLOPs

  SISO SSM state updates (outer product, AI ~2) sit in memory-bound territory.
  MIMO SSM state updates (matmul rank-r, AI ~2r) shift toward compute-bound.

  On CPU (ridge ~0.5), even SISO is near the ridge — the effect is modest.
  On GPU (ridge ~300), SISO uses <1% of the chip. MIMO is essential.

  This is the hardware motivation for Mamba-3's MIMO formulation:
  not just better models, but models that actually use the silicon.
""")

    total_time = time.time() - total_start
    print(f"Total runtime: {total_time:.1f}s")


if __name__ == "__main__":
    main()
