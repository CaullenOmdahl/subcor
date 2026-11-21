"""Function-preserving dynamic low-rank expansion and quantized (STE) adaptation.

Implements an ElasticLinear layer wrapping a strictly immutable (read-only)
base weight matrix W_base plus an ordered stack of hot-swappable low-rank
adapters. New capacity can be appended dynamically (r -> r + k) with exact
zero output perturbation, then adapted via either FP32 gradients or Binary
Straight-Through Estimation (STE) while keeping W_base and prior adapters
completely frozen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from subcor.sub1bit import Sub1BitMatrix, _pack_pm1, factorize_sub1bit


def _sign_pm1(arr: np.ndarray) -> np.ndarray:
    return np.where(arr >= 0.0, 1.0, -1.0)


@dataclass
class LowRankAdapter:
    """Hot-swappable low-rank weight delta: Delta W = scale * (B @ A).

    For FP32 adapters:
      - A has shape (rank, d_in)
      - B has shape (d_out, rank)
      - Initialized with B = 0 so initial Delta W == 0 exactly.

    For Binary STE adapters (binary_ste=True):
      - Matches Sub1BitMatrix parameterization directly:
            Delta W = scale * diag(s_out) @ (sign(B) @ diag(sigma) @ sign(A)) @ diag(s_in)
      - s_out is initialized to 0 so initial Delta W == 0 exactly.
      - Forward pass uses sign(B) and sign(A) with real scales; backward pass uses
        clipped Straight-Through Estimator (|z| <= 1), so exporting to Sub1BitMatrix
        incurs zero post-training quantization cliff.
    """

    name: str
    a: np.ndarray  # (rank, d_in)
    b: np.ndarray  # (d_out, rank)
    scale: float = 1.0
    binary_ste: bool = False
    s_out: np.ndarray | None = None  # (d_out,) used when binary_ste=True
    s_in: np.ndarray | None = None  # (d_in,) used when binary_ste=True
    sigma: np.ndarray | None = None  # (rank,) used when binary_ste=True
    frozen: bool = False

    def __post_init__(self) -> None:
        self.a = np.asarray(self.a, dtype=np.float64)
        self.b = np.asarray(self.b, dtype=np.float64)
        if self.a.ndim != 2 or self.b.ndim != 2:
            raise ValueError("Adapter factors A and B must be 2D matrices")
        if self.a.shape[0] != self.b.shape[1]:
            raise ValueError(f"Rank mismatch: B has {self.b.shape[1]}, A has {self.a.shape[0]}")
        if self.binary_ste:
            if self.s_out is None:
                self.s_out = np.zeros(self.b.shape[0], dtype=np.float64)
            else:
                self.s_out = np.asarray(self.s_out, dtype=np.float64)
            if self.s_in is None:
                self.s_in = np.ones(self.a.shape[1], dtype=np.float64)
            else:
                self.s_in = np.asarray(self.s_in, dtype=np.float64)
            if self.sigma is None:
                init_sig = 1.0 / np.sqrt(self.a.shape[1] * self.a.shape[0])
                self.sigma = np.full(self.a.shape[0], init_sig, dtype=np.float64)
            else:
                self.sigma = np.asarray(self.sigma, dtype=np.float64)

    @property
    def d_out(self) -> int:
        return self.b.shape[0]

    @property
    def d_in(self) -> int:
        return self.a.shape[1]

    @property
    def rank(self) -> int:
        return self.a.shape[0]

    @classmethod
    def zero_initialized(
        cls,
        name: str,
        d_out: int,
        d_in: int,
        rank: int,
        *,
        binary_ste: bool = False,
        seed: int = 0,
        scale: float = 1.0,
    ) -> LowRankAdapter:
        """Create a function-preserving adapter whose initial delta is identically zero."""
        rng = np.random.default_rng(seed)
        if binary_ste:
            # Latent weights in [-0.4, 0.4] stay well inside the active STE region |z| <= 1.
            a = rng.uniform(-0.4, 0.4, size=(rank, d_in))
            b = rng.uniform(-0.4, 0.4, size=(d_out, rank))
            s_out = np.zeros(d_out, dtype=np.float64)
            s_in = np.ones(d_in, dtype=np.float64)
            sigma = np.full(rank, 1.0 / np.sqrt(d_in * rank), dtype=np.float64)
            return cls(
                name=name,
                a=a,
                b=b,
                scale=scale,
                binary_ste=True,
                s_out=s_out,
                s_in=s_in,
                sigma=sigma,
            )
        a = rng.normal(0.0, 1.0 / np.sqrt(d_in), size=(rank, d_in))
        b = np.zeros((d_out, rank), dtype=np.float64)
        return cls(name=name, a=a, b=b, scale=scale, binary_ste=False)

    @classmethod
    def from_sub1bit(cls, name: str, sub1bit: Sub1BitMatrix) -> LowRankAdapter:
        """Mount a bit-packed Sub1BitMatrix as a frozen hot-swappable adapter."""
        u, v = sub1bit.unpack_factors()
        s1 = sub1bit.s1.astype(np.float64)
        s2 = sub1bit.s2.astype(np.float64)
        sigma = sub1bit.sigma.astype(np.float64)
        b = (s1[:, np.newaxis] * u) * sigma[np.newaxis, :]
        a = v.T * s2[np.newaxis, :]
        return cls(name=name, a=a, b=b, scale=1.0, binary_ste=False, frozen=True)

    def delta_matrix(self) -> np.ndarray:
        """Materialize effective Delta W of shape (d_out, d_in)."""
        if self.binary_ste:
            assert self.s_out is not None and self.s_in is not None and self.sigma is not None
            b_q = _sign_pm1(self.b)
            a_q = _sign_pm1(self.a)
            core = (b_q * self.sigma[np.newaxis, :]) @ a_q
            return self.scale * (self.s_out[:, np.newaxis] * core) * self.s_in[np.newaxis, :]
        return self.scale * (self.b @ self.a)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute adapter contribution for batch x of shape (..., d_in)."""
        if self.binary_ste:
            assert self.s_out is not None and self.s_in is not None and self.sigma is not None
            a_q = _sign_pm1(self.a)
            b_q = _sign_pm1(self.b)
            h = ((x * self.s_in) @ a_q.T) * self.sigma
            return self.scale * ((h @ b_q.T) * self.s_out)
        return self.scale * ((x @ self.a.T) @ self.b.T)

    def expand_rank(self, additional_rank: int, *, seed: int = 0) -> None:
        """Dynamically grow rank r -> r + k with strict zero output perturbation."""
        if self.frozen:
            raise RuntimeError(f"Cannot expand frozen adapter '{self.name}'")
        if additional_rank <= 0:
            raise ValueError("additional_rank must be positive")
        if self.binary_ste:
            raise RuntimeError(
                "For binary_ste adapters, append a new zero-initialized adapter branch "
                "on ElasticLinear to guarantee exact zero perturbation across shared scales."
            )
        rng = np.random.default_rng(seed)
        a_new = rng.normal(0.0, 1.0 / np.sqrt(self.d_in), size=(additional_rank, self.d_in))
        b_new = np.zeros((self.d_out, additional_rank), dtype=np.float64)
        self.a = np.vstack([self.a, a_new])
        self.b = np.hstack([self.b, b_new])

    def compress_sub1bit(self, *, rank: int | None = None) -> Sub1BitMatrix:
        """Compress this adapter's effective delta into a Sub1BitMatrix.

        When the adapter was trained with Binary STE at the target rank, its
        already-quantized binary factors and scaling vectors are packed directly
        without post-training re-factorization loss.
        """
        target_rank = self.rank if rank is None else rank
        if self.binary_ste and target_rank == self.rank:
            assert self.s_out is not None and self.s_in is not None and self.sigma is not None
            u_bin = _sign_pm1(self.b)  # (d_out, rank)
            v_bin = _sign_pm1(self.a.T)  # (d_in, rank)
            return Sub1BitMatrix(
                d_out=self.d_out,
                d_in=self.d_in,
                rank=self.rank,
                s1=self.s_out.astype(np.float16),
                s2=self.s_in.astype(np.float16),
                sigma=(self.scale * self.sigma).astype(np.float16),
                u_packed=_pack_pm1(u_bin),
                v_packed=_pack_pm1(v_bin),
            )
        return factorize_sub1bit(self.delta_matrix(), rank=target_rank)


class ElasticLinear:
    """Linear layer with strictly frozen base weights + hot-swappable low-rank adapters."""

    def __init__(self, weight: np.ndarray, bias: np.ndarray | None = None) -> None:
        base = np.array(weight, dtype=np.float64, copy=True)
        if base.ndim != 2:
            raise ValueError(f"Expected 2D base weight matrix, got shape {base.shape}")
        base.setflags(write=False)
        self._w_base = base

        if bias is not None:
            b = np.array(bias, dtype=np.float64, copy=True)
            if b.shape != (base.shape[0],):
                raise ValueError("Bias shape must match d_out")
            b.setflags(write=False)
            self._bias: np.ndarray | None = b
        else:
            self._bias = None

        self._adapters: dict[str, LowRankAdapter] = {}

    @property
    def d_out(self) -> int:
        return self._w_base.shape[0]

    @property
    def d_in(self) -> int:
        return self._w_base.shape[1]

    @property
    def w_base(self) -> np.ndarray:
        """Read-only view of the immutable base weight matrix."""
        return self._w_base

    @property
    def bias(self) -> np.ndarray | None:
        return self._bias

    @property
    def adapter_names(self) -> tuple[str, ...]:
        return tuple(self._adapters.keys())

    def mount_adapter(self, adapter: LowRankAdapter) -> None:
        """Attach or replace a named adapter without modifying w_base."""
        if adapter.d_out != self.d_out or adapter.d_in != self.d_in:
            raise ValueError(
                f"Adapter shape ({adapter.d_out}, {adapter.d_in}) does not match "
                f"layer shape ({self.d_out}, {self.d_in})"
            )
        self._adapters[adapter.name] = adapter

    def unmount_adapter(self, name: str) -> LowRankAdapter | None:
        """Remove a named adapter atomically (instant rollback)."""
        return self._adapters.pop(name, None)

    def get_adapter(self, name: str) -> LowRankAdapter:
        return self._adapters[name]

    def expand(
        self,
        name: str,
        rank: int,
        *,
        binary_ste: bool = False,
        seed: int = 0,
        scale: float = 1.0,
    ) -> LowRankAdapter:
        """Append a new zero-initialized adapter branch (strictly function-preserving)."""
        if name in self._adapters:
            raise ValueError(f"Adapter '{name}' is already mounted")
        adapter = LowRankAdapter.zero_initialized(
            name=name,
            d_out=self.d_out,
            d_in=self.d_in,
            rank=rank,
            binary_ste=binary_ste,
            seed=seed,
            scale=scale,
        )
        self._adapters[name] = adapter
        return adapter

    def effective_weight(self) -> np.ndarray:
        """Return effective W_base + sum(adapter deltas) for inspection/diagnostics."""
        w = self._w_base.copy()
        for adapter in self._adapters.values():
            w += adapter.delta_matrix()
        return w

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute y = x @ W_base.T + bias + sum_m adapter_m(x)."""
        x_arr = np.asarray(x, dtype=np.float64)
        out = x_arr @ self._w_base.T
        if self._bias is not None:
            out = out + self._bias
        for adapter in self._adapters.values():
            out = out + adapter.forward(x_arr)
        return out

    def fit_adapter_mse(
        self,
        adapter_name: str,
        x: np.ndarray,
        y_target: np.ndarray,
        *,
        steps: int = 200,
        lr: float = 0.05,
        l2_reg: float = 1e-4,
    ) -> list[float]:
        """Adapt ONLY the named adapter to match y_target via Adam (FP32 or Binary STE).

        Base weights and all other mounted adapters remain strictly frozen.
        Returns the per-step MSE loss curve.
        """
        adapter = self._adapters[adapter_name]
        if adapter.frozen:
            raise RuntimeError(f"Adapter '{adapter_name}' is frozen")

        x_arr = np.asarray(x, dtype=np.float64)
        y_arr = np.asarray(y_target, dtype=np.float64)
        n = x_arr.shape[0]

        # Precompute fixed output from frozen base + all other adapters.
        fixed_out = x_arr @ self._w_base.T
        if self._bias is not None:
            fixed_out = fixed_out + self._bias
        for other_name, other in self._adapters.items():
            if other_name != adapter_name:
                fixed_out = fixed_out + other.forward(x_arr)

        residual_target = y_arr - fixed_out
        losses: list[float] = []

        if not adapter.binary_ste:
            m_a = np.zeros_like(adapter.a)
            v_a = np.zeros_like(adapter.a)
            m_b = np.zeros_like(adapter.b)
            v_b = np.zeros_like(adapter.b)
            beta1, beta2, eps = 0.9, 0.999, 1e-8

            for t in range(1, steps + 1):
                h = x_arr @ adapter.a.T  # (n, rank)
                pred = adapter.scale * (h @ adapter.b.T)  # (n, d_out)
                err = pred - residual_target
                loss = float(np.mean(err * err))
                losses.append(loss)

                d_pred = (2.0 / (n * self.d_out)) * err * adapter.scale  # (n, d_out)
                grad_b = d_pred.T @ h + l2_reg * adapter.b
                grad_h = d_pred @ adapter.b  # (n, rank)
                grad_a = grad_h.T @ x_arr + l2_reg * adapter.a

                m_a = beta1 * m_a + (1.0 - beta1) * grad_a
                v_a = beta2 * v_a + (1.0 - beta2) * (grad_a * grad_a)
                m_b = beta1 * m_b + (1.0 - beta1) * grad_b
                v_b = beta2 * v_b + (1.0 - beta2) * (grad_b * grad_b)

                m_a_hat = m_a / (1.0 - beta1**t)
                v_a_hat = v_a / (1.0 - beta2**t)
                m_b_hat = m_b / (1.0 - beta1**t)
                v_b_hat = v_b / (1.0 - beta2**t)

                adapter.a -= lr * m_a_hat / (np.sqrt(v_a_hat) + eps)
                adapter.b -= lr * m_b_hat / (np.sqrt(v_b_hat) + eps)
            return losses

        # Binary Straight-Through Estimator (STE) QAT training directly in {-1, +1} space.
        assert adapter.s_out is not None and adapter.s_in is not None and adapter.sigma is not None
        params = [adapter.a, adapter.b, adapter.s_out, adapter.s_in, adapter.sigma]
        m_state = [np.zeros_like(p) for p in params]
        v_state = [np.zeros_like(p) for p in params]
        beta1, beta2, eps = 0.9, 0.999, 1e-8

        for t in range(1, steps + 1):
            # Cosine decay helps binary sign flips settle cleanly in later steps.
            step_lr = lr * (0.15 + 0.85 * 0.5 * (1.0 + np.cos(np.pi * (t - 1) / max(1, steps))))

            a_q = _sign_pm1(adapter.a)
            b_q = _sign_pm1(adapter.b)
            x_scaled = x_arr * adapter.s_in[np.newaxis, :]  # (n, d_in)
            h_raw = x_scaled @ a_q.T  # (n, rank)
            h = h_raw * adapter.sigma[np.newaxis, :]  # (n, rank)
            z = h @ b_q.T  # (n, d_out)
            pred = adapter.scale * (z * adapter.s_out[np.newaxis, :])

            err = pred - residual_target
            loss = float(np.mean(err * err))
            losses.append(loss)

            d_pred = (2.0 / (n * self.d_out)) * err * adapter.scale  # (n, d_out)
            grad_s_out = np.sum(d_pred * z, axis=0) + l2_reg * adapter.s_out
            d_z = d_pred * adapter.s_out[np.newaxis, :]  # (n, d_out)

            grad_bq = d_z.T @ h  # (d_out, rank)
            d_h = d_z @ b_q  # (n, rank)
            grad_sigma = np.sum(d_h * h_raw, axis=0) + l2_reg * adapter.sigma
            d_h_raw = d_h * adapter.sigma[np.newaxis, :]  # (n, rank)
            grad_aq = d_h_raw.T @ x_scaled  # (rank, d_in)
            d_x_scaled = d_h_raw @ a_q  # (n, d_in)
            grad_s_in = np.sum(d_x_scaled * x_arr, axis=0) + l2_reg * adapter.s_in

            # Clipped STE: pass gradients through sign() only where |latent| <= 1.
            grad_a = grad_aq * (np.abs(adapter.a) <= 1.0)
            grad_b = grad_bq * (np.abs(adapter.b) <= 1.0)

            grads = [grad_a, grad_b, grad_s_out, grad_s_in, grad_sigma]
            for idx, (param, grad) in enumerate(zip(params, grads, strict=True)):
                m_state[idx] = beta1 * m_state[idx] + (1.0 - beta1) * grad
                v_state[idx] = beta2 * v_state[idx] + (1.0 - beta2) * (grad * grad)
                m_hat = m_state[idx] / (1.0 - beta1**t)
                v_hat = v_state[idx] / (1.0 - beta2**t)
                param -= step_lr * m_hat / (np.sqrt(v_hat) + eps)

            # Keep latent weights bounded inside [-1.0, 1.0] so sign flips remain responsive.
            np.clip(adapter.a, -1.0, 1.0, out=adapter.a)
            np.clip(adapter.b, -1.0, 1.0, out=adapter.b)

        return losses
