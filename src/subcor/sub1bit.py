"""Sub-1-bit low-rank binary matrix factorization and bit-packed serialization.

Inspired by NanoQuant (arXiv:2602.06694): instead of storing >= 1 bit per
discrete parameter in a dense d_out x d_in weight matrix, we factorize:

    W ≈ diag(s1) @ (U_bin @ diag(sigma) @ V_bin.T) @ diag(s2)

where U_bin ∈ {-1, +1}^{d_out x r}, V_bin ∈ {-1, +1}^{d_in x r} are stored as
bit-packed uint8 buffers (1 bit per entry) and s1, s2, sigma are float16
scaling vectors. Whenever r * (d_out + d_in) + 16 * (d_out + d_in + r) <
d_out * d_in, average storage drops strictly below 1 bit per parameter.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

import numpy as np

_MAGIC = b"SC1B"
_VERSION = 1
_HEADER_STRUCT = struct.Struct("<4sBIII")  # magic, version, d_out, d_in, rank


def _sign_pm1(arr: np.ndarray) -> np.ndarray:
    """Map real values to {-1.0, +1.0}, mapping 0.0 -> +1.0 deterministically."""
    return np.where(arr >= 0.0, 1.0, -1.0)


def _pack_pm1(mat: np.ndarray) -> bytes:
    """Bit-pack a {-1, +1} matrix into compact uint8 bytes."""
    bits = (mat.reshape(-1) > 0).astype(np.uint8)
    return np.packbits(bits, bitorder="little").tobytes()


def _unpack_pm1(raw: bytes, shape: tuple[int, int]) -> np.ndarray:
    """Unpack bit-packed bytes back into a float64 {-1.0, +1.0} matrix."""
    total = shape[0] * shape[1]
    packed = np.frombuffer(raw, dtype=np.uint8)
    bits = np.unpackbits(packed, bitorder="little")[:total]
    return np.where(bits > 0, 1.0, -1.0).reshape(shape)


@dataclass(frozen=True)
class Sub1BitMatrix:
    """Bit-packed low-rank binary matrix representation."""

    d_out: int
    d_in: int
    rank: int
    s1: np.ndarray  # float16 shape (d_out,)
    s2: np.ndarray  # float16 shape (d_in,)
    sigma: np.ndarray  # float16 shape (rank,)
    u_packed: bytes
    v_packed: bytes

    def __post_init__(self) -> None:
        if self.d_out <= 0 or self.d_in <= 0 or self.rank <= 0:
            raise ValueError("d_out, d_in, and rank must be positive integers")
        expected_u = math.ceil((self.d_out * self.rank) / 8)
        expected_v = math.ceil((self.d_in * self.rank) / 8)
        if len(self.u_packed) != expected_u:
            raise ValueError(f"Expected {expected_u} packed U bytes, got {len(self.u_packed)}")
        if len(self.v_packed) != expected_v:
            raise ValueError(f"Expected {expected_v} packed V bytes, got {len(self.v_packed)}")
        if self.s1.shape != (self.d_out,) or self.s2.shape != (self.d_in,):
            raise ValueError("Scale vector shapes do not match matrix dimensions")
        if self.sigma.shape != (self.rank,):
            raise ValueError("Sigma vector shape does not match rank")

    @property
    def shape(self) -> tuple[int, int]:
        return (self.d_out, self.d_in)

    @property
    def num_weights(self) -> int:
        return self.d_out * self.d_in

    @property
    def payload_bits(self) -> int:
        """Exact stored parameter bits (binary factors + float16 scaling vectors)."""
        binary_bits = (self.d_out + self.d_in) * self.rank
        scale_bits = 16 * (self.d_out + self.d_in + self.rank)
        return binary_bits + scale_bits

    @property
    def bits_per_weight(self) -> float:
        """Effective parameter bits per original dense weight."""
        return self.payload_bits / self.num_weights

    @property
    def serialized_bits_per_weight(self) -> float:
        """Total serialized wire bits (including header and byte alignment) per weight."""
        return (len(self.to_bytes()) * 8) / self.num_weights

    def unpack_factors(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (U_pm1, V_pm1) in {-1.0, +1.0}."""
        u = _unpack_pm1(self.u_packed, (self.d_out, self.rank))
        v = _unpack_pm1(self.v_packed, (self.d_in, self.rank))
        return u, v

    def reconstruct(self) -> np.ndarray:
        """Reconstruct dense float64 approximation of W."""
        u, v = self.unpack_factors()
        s1 = self.s1.astype(np.float64)
        s2 = self.s2.astype(np.float64)
        sigma = self.sigma.astype(np.float64)
        core = (u * sigma[np.newaxis, :]) @ v.T
        return (s1[:, np.newaxis] * core) * s2[np.newaxis, :]

    def matvec(self, x: np.ndarray) -> np.ndarray:
        """Apply reconstructed matrix to row-batch x of shape (..., d_in) without materializing W."""
        u, v = self.unpack_factors()
        s1 = self.s1.astype(np.float64)
        s2 = self.s2.astype(np.float64)
        sigma = self.sigma.astype(np.float64)
        scaled_x = x * s2
        proj = scaled_x @ v  # (..., rank)
        out = (proj * sigma) @ u.T  # (..., d_out)
        return out * s1

    def to_bytes(self) -> bytes:
        """Serialize to deterministic self-describing binary format."""
        header = _HEADER_STRUCT.pack(_MAGIC, _VERSION, self.d_out, self.d_in, self.rank)
        return b"".join(
            [
                header,
                self.s1.astype("<f2", copy=False).tobytes(),
                self.s2.astype("<f2", copy=False).tobytes(),
                self.sigma.astype("<f2", copy=False).tobytes(),
                self.u_packed,
                self.v_packed,
            ]
        )

    def sha256(self) -> str:
        """Content-addressed SHA-256 digest of serialized bytes."""
        return hashlib.sha256(self.to_bytes()).hexdigest()

    @classmethod
    def from_bytes(cls, data: bytes) -> Sub1BitMatrix:
        """Deserialize from bytes created by to_bytes()."""
        if len(data) < _HEADER_STRUCT.size:
            raise ValueError("Truncated Sub1BitMatrix payload")
        magic, version, d_out, d_in, rank = _HEADER_STRUCT.unpack_from(data, 0)
        if magic != _MAGIC:
            raise ValueError(f"Invalid Sub1BitMatrix magic: {magic!r}")
        if version != _VERSION:
            raise ValueError(f"Unsupported Sub1BitMatrix version: {version}")

        offset = _HEADER_STRUCT.size

        def take(nbytes: int) -> bytes:
            nonlocal offset
            end = offset + nbytes
            if end > len(data):
                raise ValueError("Truncated Sub1BitMatrix body")
            chunk = data[offset:end]
            offset = end
            return chunk

        s1 = np.frombuffer(take(d_out * 2), dtype="<f2").astype(np.float16, copy=True)
        s2 = np.frombuffer(take(d_in * 2), dtype="<f2").astype(np.float16, copy=True)
        sigma = np.frombuffer(take(rank * 2), dtype="<f2").astype(np.float16, copy=True)
        u_bytes = take(math.ceil((d_out * rank) / 8))
        v_bytes = take(math.ceil((d_in * rank) / 8))

        if offset != len(data):
            raise ValueError(f"Unexpected trailing bytes in Sub1BitMatrix: {len(data) - offset}")

        return cls(
            d_out=d_out,
            d_in=d_in,
            rank=rank,
            s1=s1,
            s2=s2,
            sigma=sigma,
            u_packed=u_bytes,
            v_packed=v_bytes,
        )


def _leading_singular_vectors(
    mat: np.ndarray, *, power_iters: int = 12, eps: float = 1e-12
) -> tuple[np.ndarray, np.ndarray]:
    """Compute leading singular pair (u, v) via power iteration for large matrices."""
    d_out, d_in = mat.shape
    if min(d_out, d_in) <= 96:
        u_svd, _, vt_svd = np.linalg.svd(mat, full_matrices=False)
        return u_svd[:, 0], vt_svd[0, :]

    # Deterministic power iteration avoids O(min(m, n)^2 * max(m, n)) full SVD cost.
    v = np.mean(mat, axis=0)
    v_norm = float(np.linalg.norm(v))
    v = np.ones(d_in, dtype=np.float64) / math.sqrt(d_in) if v_norm < eps else v / v_norm

    u = np.ones(d_out, dtype=np.float64) / math.sqrt(d_out)
    for _ in range(power_iters):
        u_next = mat @ v
        u_norm = float(np.linalg.norm(u_next))
        if u_norm < eps:
            break
        u = u_next / u_norm
        v_next = mat.T @ u
        v_norm = float(np.linalg.norm(v_next))
        if v_norm < eps:
            break
        v = v_next / v_norm
    return u, v


def _best_binary_rank1(
    residual: np.ndarray, *, sign_iters: int = 8
) -> tuple[np.ndarray, np.ndarray, float]:
    """Find (u, v, sigma) with u, v in {-1, +1} approximating residual ≈ sigma * u v^T."""
    d_out, d_in = residual.shape
    norm = float(np.linalg.norm(residual))
    if norm < 1e-12:
        return np.ones(d_out, dtype=np.float64), np.ones(d_in, dtype=np.float64), 0.0

    # Initialize from leading singular vectors of the residual.
    u_lead, v_lead = _leading_singular_vectors(residual)
    u = _sign_pm1(u_lead)
    v = _sign_pm1(v_lead)

    # Alternating maximization of u^T R v over {-1, +1} monotonically increases objective.
    for _ in range(sign_iters):
        u_next = _sign_pm1(residual @ v)
        v_next = _sign_pm1(residual.T @ u_next)
        if np.array_equal(u_next, u) and np.array_equal(v_next, v):
            break
        u, v = u_next, v_next

    sigma = float((u @ residual @ v) / (d_out * d_in))
    return u, v, sigma


def factorize_sub1bit(
    weight: np.ndarray,
    *,
    rank: int,
    refinement_sweeps: int = 10,
    eps: float = 1e-12,
) -> Sub1BitMatrix:
    """Factorize a 2D weight matrix into a bit-packed Sub1BitMatrix.

    Algorithm:
      1. Greedy residual binary rank-1 decomposition for rank components (U, V, sigma).
      2. Alternating refinement sweeps optimizing row scales (s1), column scales (s2),
         and component magnitudes (sigma) in closed form plus binary sign refinement.
    """
    w = np.asarray(weight, dtype=np.float64)
    if w.ndim != 2:
        raise ValueError(f"Expected 2D weight matrix, got shape {w.shape}")
    d_out, d_in = w.shape
    if rank <= 0 or rank > min(d_out, d_in):
        raise ValueError(f"rank must be in [1, {min(d_out, d_in)}], got {rank}")

    if float(np.linalg.norm(w)) < eps:
        u_ones = np.ones((d_out, rank), dtype=np.float64)
        v_ones = np.ones((d_in, rank), dtype=np.float64)
        return Sub1BitMatrix(
            d_out=d_out,
            d_in=d_in,
            rank=rank,
            s1=np.zeros(d_out, dtype=np.float16),
            s2=np.zeros(d_in, dtype=np.float16),
            sigma=np.zeros(rank, dtype=np.float16),
            u_packed=_pack_pm1(u_ones),
            v_packed=_pack_pm1(v_ones),
        )

    # Initial row/col RMS normalization makes binary factors well-conditioned.
    row_rms = np.sqrt(np.mean(w * w, axis=1) + eps)
    w_row_norm = w / row_rms[:, np.newaxis]
    col_rms = np.sqrt(np.mean(w_row_norm * w_row_norm, axis=0) + eps)
    w_norm = w_row_norm / col_rms[np.newaxis, :]

    s1 = row_rms.copy()
    s2 = col_rms.copy()

    u = np.ones((d_out, rank), dtype=np.float64)
    v = np.ones((d_in, rank), dtype=np.float64)
    sigma = np.zeros(rank, dtype=np.float64)

    residual = w_norm.copy()
    for k in range(rank):
        uk, vk, sk = _best_binary_rank1(residual)
        u[:, k] = uk
        v[:, k] = vk
        sigma[k] = sk
        residual = residual - sk * np.outer(uk, vk)

    # Alternating closed-form scale refinement and binary sign updates.
    for _ in range(refinement_sweeps):
        # 1. Refine each rank-k binary component against the weighted residual.
        core = (u * sigma[np.newaxis, :]) @ v.T
        for k in range(rank):
            core_without_k = core - sigma[k] * np.outer(u[:, k], v[:, k])
            approx_without_k = (s1[:, np.newaxis] * core_without_k) * s2[np.newaxis, :]
            err = w - approx_without_k
            # Weighted objective: minimize ||err - s1 * sigma_k * (u_k v_k^T) * s2^T||_F^2
            # Equivalent to maximizing (u_k * s1)^T err (v_k * s2) up to scale.
            uk = u[:, k]
            vk = v[:, k]
            for _ in range(3):
                uk_new = _sign_pm1(s1 * (err @ (s2 * vk)))
                vk_new = _sign_pm1(s2 * (err.T @ (s1 * uk_new)))
                if np.array_equal(uk_new, uk) and np.array_equal(vk_new, vk):
                    break
                uk, vk = uk_new, vk_new
            denom = float(np.sum((s1 * uk) ** 2) * np.sum((s2 * vk) ** 2))
            sk = float(((s1 * uk) @ err @ (s2 * vk)) / (denom + eps))
            u[:, k] = uk
            v[:, k] = vk
            sigma[k] = sk
            core = core_without_k + sk * np.outer(uk, vk)

        # 2. Closed-form least-squares update for row scaling s1.
        m_col = core * s2[np.newaxis, :]
        denom_s1 = np.sum(m_col * m_col, axis=1)
        s1 = np.where(denom_s1 > eps, np.sum(w * m_col, axis=1) / (denom_s1 + eps), 0.0)

        # 3. Closed-form least-squares update for column scaling s2.
        m_row = s1[:, np.newaxis] * core
        denom_s2 = np.sum(m_row * m_row, axis=0)
        s2 = np.where(denom_s2 > eps, np.sum(w * m_row, axis=0) / (denom_s2 + eps), 0.0)

        # Re-normalize scales into balanced magnitude to avoid float16 overflow/underflow.
        s1_norm = float(np.sqrt(np.mean(s1 * s1) + eps))
        s2_norm = float(np.sqrt(np.mean(s2 * s2) + eps))
        if s1_norm > eps and s2_norm > eps:
            s1 = s1 / s1_norm
            s2 = s2 / s2_norm
            sigma = sigma * (s1_norm * s2_norm)

    # Guard float16 finite range by absorbing global scale cleanly across s1, s2, sigma.
    max_sigma = float(np.max(np.abs(sigma)))
    if max_sigma > 1e3:
        factor = math.sqrt(max_sigma / 10.0)
        sigma = sigma / (factor * factor)
        s1 = s1 * factor
        s2 = s2 * factor

    return Sub1BitMatrix(
        d_out=d_out,
        d_in=d_in,
        rank=rank,
        s1=s1.astype(np.float16),
        s2=s2.astype(np.float16),
        sigma=sigma.astype(np.float16),
        u_packed=_pack_pm1(u),
        v_packed=_pack_pm1(v),
    )


def relative_frobenius_error(original: np.ndarray, factorized: Sub1BitMatrix) -> float:
    """Return ||W - W_hat||_F / ||W||_F (0.0 if both are zero)."""
    w = np.asarray(original, dtype=np.float64)
    w_hat = factorized.reconstruct()
    denom = float(np.linalg.norm(w))
    if denom < 1e-12:
        return 0.0 if float(np.linalg.norm(w_hat)) < 1e-12 else 1.0
    return float(np.linalg.norm(w - w_hat) / denom)
