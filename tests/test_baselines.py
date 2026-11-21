"""Tests for Issue #12 controlled baselines and float16 denormal protection."""

from __future__ import annotations

import numpy as np

from subcor.baselines import (
    FrozenEmbeddingKNNBaseline,
    compare_compression_at_equal_bytes,
    generate_heldout_procedural_suite,
    run_router_scale_sweep,
)
from subcor.sub1bit import factorize_sub1bit, relative_frobenius_error


def test_sub1bit_preserves_tiny_deltas_without_fp16_denormal_underflow() -> None:
    """Issue #12: 1e-6 magnitude deltas must be strictly scale-invariant with normal fp16 scales."""
    rng = np.random.default_rng(12)
    u = rng.choice([-1.0, 1.0], size=(64, 4))
    v = rng.choice([-1.0, 1.0], size=(64, 4))
    w_unit = u @ v.T
    w_tiny = w_unit * 1e-6

    sub_unit = factorize_sub1bit(w_unit, rank=4)
    sub_tiny = factorize_sub1bit(w_tiny, rank=4)

    err_unit = relative_frobenius_error(w_unit, sub_unit)
    err_tiny = relative_frobenius_error(w_tiny, sub_tiny)
    assert abs(err_tiny - err_unit) < 2e-3
    # All non-zero float16 scale entries stay safely above float16 normal minimum (6.1e-5).
    assert float(np.min(np.abs(sub_tiny.s1))) > 6.1e-5
    assert float(np.min(np.abs(sub_tiny.s2))) > 6.1e-5


def test_compression_equal_byte_comparison_and_router_scale_sweep() -> None:
    rows = compare_compression_at_equal_bytes(d_out=128, d_in=256, true_rank=8, seed=7)
    by_method = {r.method: r for r in rows}
    assert "sub1bit_rank_8" in by_method
    assert "fp16_lora_rank_1" in by_method
    assert "fp16_lora_rank_2" in by_method
    assert "bitdelta_dense_1bit" in by_method
    # Sub-1-bit must be dramatically smaller than dense BitDelta (1 bit/weight)
    assert (
        by_method["sub1bit_rank_8"].bits_per_weight
        < by_method["bitdelta_dense_1bit"].bits_per_weight
    )

    sweep = run_router_scale_sweep(skill_counts=(1, 5, 10), dim=64, n_ood_general=50, seed=7)
    assert len(sweep) == 6
    for row in sweep:
        assert 0.0 <= row.false_positive_rate <= 1.0
        assert 0.0 <= row.false_negative_rate <= 1.0
        assert 0.0 <= row.exact_skill_accuracy <= 1.0


def test_heldout_procedural_suite_has_zero_train_test_overlap_and_knn_works() -> None:
    train, test = generate_heldout_procedural_suite(n_train=20, n_test=100)
    assert len(train) == 20
    assert len(test) == 100
    assert {p for p, _ in train}.isdisjoint({p for p, _ in test})

    knn = FrozenEmbeddingKNNBaseline(k=3)
    embs = np.eye(4, dtype=np.float64)
    knn.fit(embs, ["a", "b", "c", "d"])
    assert knn.predict(np.array([0.9, 0.1, 0.0, 0.0])) == "a"
