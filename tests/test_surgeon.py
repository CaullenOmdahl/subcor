"""Tests for the zero-gradient Contrastive Weight-Space Surgeon."""

from __future__ import annotations

import numpy as np

from subcor.surgeon import WeightSpaceSurgeon


def test_surgeon_identifies_peak_anomaly_layer_and_ablates_direction() -> None:
    rng = np.random.default_rng(17)
    dim = 32
    q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
    true_loop_dir = q[:, 0]
    clean_signal_dir = q[:, 1]

    # layer_0 has little contrastive separation; layer_1 has strong separation along true_loop_dir.
    clean_l0 = rng.normal(0.0, 0.2, size=(40, dim))
    fail_l0 = rng.normal(0.0, 0.2, size=(40, dim)) + 0.2 * q[:, 2]

    clean_l1 = rng.normal(0.0, 0.15, size=(40, dim)) + np.outer(
        rng.uniform(0.5, 1.5, size=40), clean_signal_dir
    )
    fail_l1 = clean_l1 + np.outer(rng.uniform(2.5, 3.5, size=40), true_loop_dir)

    surgeon = WeightSpaceSurgeon()
    ranked = surgeon.rank_layers(
        {"layer_0": (fail_l0, clean_l0), "layer_1": (fail_l1, clean_l1)},
        rank=1,
    )

    assert ranked[0].layer_name == "layer_1"
    recovered = ranked[0].directions[:, 0]
    assert abs(float(np.dot(recovered, true_loop_dir))) > 0.99

    # Synthesize zero-gradient ablation patch for layer_1 weights.
    w_base = 2.0 * np.outer(true_loop_dir, q[:, 3]) + 1.5 * np.outer(clean_signal_dir, q[:, 4])
    patch = surgeon.synthesize_ablation_patch(
        ranked[0],
        w_base,
        adapter_name="ablate-loop",
        alpha=1.0,
    )

    w_patched = w_base + patch.adapter.delta_matrix()

    # Projection onto offending direction is annihilated; orthogonal signal is preserved.
    h_test = rng.normal(size=(20, dim))
    out_before = h_test @ w_base.T
    out_after = h_test @ w_patched.T

    loop_energy_before = float(np.mean((out_before @ true_loop_dir) ** 2))
    loop_energy_after = float(np.mean((out_after @ true_loop_dir) ** 2))
    clean_energy_before = float(np.mean((out_before @ clean_signal_dir) ** 2))
    clean_energy_after = float(np.mean((out_after @ clean_signal_dir) ** 2))

    assert loop_energy_before > 1.0
    assert loop_energy_after < 1e-2
    assert abs(clean_energy_after - clean_energy_before) < 0.05
