"""Tests for function-preserving dynamic low-rank expansion and STE adaptation."""

from __future__ import annotations

import numpy as np
import pytest

from subcor.expansion import ElasticLinear, LowRankAdapter


def test_frozen_base_cannot_be_mutated() -> None:
    w = np.eye(16, dtype=np.float64)
    layer = ElasticLinear(w)
    assert layer.w_base.flags.writeable is False
    with pytest.raises(ValueError):
        layer.w_base[0, 0] = 99.0


@pytest.mark.parametrize("binary_ste", [False, True])
def test_dynamic_expansion_is_strictly_zero_perturbation(binary_ste: bool) -> None:
    rng = np.random.default_rng(3)
    layer = ElasticLinear(rng.normal(size=(32, 32)))
    x = rng.normal(size=(24, 32))

    y_before = layer.forward(x)
    layer.expand("skill-branch", rank=4, binary_ste=binary_ste, seed=5)
    y_after = layer.forward(x)

    assert float(np.max(np.abs(y_after - y_before))) == 0.0


def test_fp32_adapter_expand_rank_preserves_function() -> None:
    rng = np.random.default_rng(9)
    adapter = LowRankAdapter(
        name="a1",
        a=rng.normal(size=(2, 16)),
        b=rng.normal(size=(16, 2)),
    )
    x = rng.normal(size=(10, 16))
    y_before = adapter.forward(x)
    adapter.expand_rank(3, seed=4)
    y_after = adapter.forward(x)

    assert adapter.rank == 5
    np.testing.assert_allclose(y_after, y_before, atol=1e-12)


@pytest.mark.parametrize("binary_ste", [False, True])
def test_adapter_learns_residual_while_base_and_prior_adapter_stay_frozen(binary_ste: bool) -> None:
    rng = np.random.default_rng(21)
    d = 32
    w_base = rng.normal(0.0, 0.3, size=(d, d))
    layer = ElasticLinear(w_base)
    w_base_snapshot = layer.w_base.copy()

    # Mount a frozen prior adapter to verify it is undisturbed.
    prior = LowRankAdapter(
        name="prior-frozen",
        a=rng.normal(0.0, 0.1, size=(1, d)),
        b=rng.normal(0.0, 0.1, size=(d, 1)),
        frozen=True,
    )
    prior_delta_before = prior.delta_matrix().copy()
    layer.mount_adapter(prior)

    # Create a rank-2 target residual mapping.
    true_u = rng.normal(0.0, 0.6, size=(d, 2))
    true_v = rng.normal(0.0, 0.6, size=(2, d))
    x = rng.normal(size=(96, d))
    y_target = layer.forward(x) + x @ (true_u @ true_v).T

    layer.expand("new-skill", rank=4, binary_ste=binary_ste, seed=13)
    losses = layer.fit_adapter_mse("new-skill", x, y_target, steps=260, lr=0.05)

    # FP32 reaches near-exact fit; Binary STE learns directly in {-1, +1} quantized space.
    max_ratio = 0.05 if not binary_ste else 0.20
    assert losses[-1] < losses[0] * max_ratio
    np.testing.assert_array_equal(layer.w_base, w_base_snapshot)
    np.testing.assert_array_equal(prior.delta_matrix(), prior_delta_before)

    if binary_ste:
        # QAT-trained Binary STE adapter packs directly into Sub1BitMatrix with no quantization cliff.
        adapter = layer.get_adapter("new-skill")
        packed = adapter.compress_sub1bit()
        np.testing.assert_allclose(packed.reconstruct(), adapter.delta_matrix(), atol=5e-3)

    # Unmounting restores exact pre-expansion output.
    y_with_skill = layer.forward(x)
    layer.unmount_adapter("new-skill")
    y_rolled_back = layer.forward(x)
    assert float(np.max(np.abs(y_with_skill - y_rolled_back))) > 0.1
    np.testing.assert_allclose(y_rolled_back, x @ w_base_snapshot.T + prior.forward(x), atol=1e-12)
