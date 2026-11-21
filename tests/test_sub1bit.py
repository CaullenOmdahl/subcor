"""Tests for sub-1-bit low-rank binary matrix factorization."""

from __future__ import annotations

import numpy as np
import pytest

from subcor.sub1bit import Sub1BitMatrix, factorize_sub1bit, relative_frobenius_error


def test_sub1bit_achieves_strictly_below_one_bit_per_weight() -> None:
    rng = np.random.default_rng(7)
    # Rank-4 structured weight matrix (256 x 256 -> 65,536 weights)
    u_true = rng.choice([-1.0, 1.0], size=(256, 4))
    v_true = rng.choice([-1.0, 1.0], size=(256, 4))
    s1_true = rng.uniform(0.5, 2.0, size=256)
    s2_true = rng.uniform(0.5, 2.0, size=256)
    w = (s1_true[:, None] * (u_true @ v_true.T)) * s2_true[None, :]

    fac = factorize_sub1bit(w, rank=4)

    assert fac.bits_per_weight < 0.25
    assert fac.serialized_bits_per_weight < 0.25
    assert relative_frobenius_error(w, fac) < 0.15


def test_sub1bit_matvec_matches_reconstruct() -> None:
    rng = np.random.default_rng(11)
    a = rng.normal(size=(64, 3))
    b = rng.normal(size=(3, 64))
    w = a @ b

    fac = factorize_sub1bit(w, rank=3)
    x = rng.normal(size=(16, 64))

    y_direct = x @ fac.reconstruct().T
    y_matvec = fac.matvec(x)
    np.testing.assert_allclose(y_matvec, y_direct, atol=1e-10)
    assert fac.bits_per_weight < 1.0


def test_sub1bit_roundtrip_serialization_and_sha256() -> None:
    rng = np.random.default_rng(19)
    w = rng.normal(size=(64, 64)) @ rng.normal(size=(64, 64)) / 64.0
    fac = factorize_sub1bit(w, rank=4)

    raw = fac.to_bytes()
    restored = Sub1BitMatrix.from_bytes(raw)

    assert restored.sha256() == fac.sha256()
    np.testing.assert_array_equal(restored.reconstruct(), fac.reconstruct())


def test_sub1bit_handles_zero_matrix_cleanly() -> None:
    w = np.zeros((32, 32), dtype=np.float64)
    fac = factorize_sub1bit(w, rank=2)
    assert relative_frobenius_error(w, fac) == 0.0
    np.testing.assert_array_equal(fac.reconstruct(), w)


def test_sub1bit_rejects_corrupt_header() -> None:
    with pytest.raises(ValueError, match="magic"):
        Sub1BitMatrix.from_bytes(b"NOPE" + b"\x00" * 32)
