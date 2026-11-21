"""Tests for P2 Issues #5 (multi-projection), #6 (column-targeted multi-layer surgery), and #14 (mlx-lm export & CLI)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from subcor.cli import main as cli_main
from subcor.proving_ground import run_proving_ground
from subcor.server import build_dummy_delta
from subcor.surgeon import WeightSpaceSurgeon
from subcor.types import AdaptationMode, GovernanceError
from subcor.units import SkillUnitRegistry


def test_column_targeted_and_multilayer_surgery() -> None:
    rng = np.random.default_rng(11)
    dim = 32
    clean_1 = rng.normal(0.0, 0.1, size=(16, dim))
    fail_dir = np.eye(dim)[0]
    fail_1 = clean_1 + 3.0 * fail_dir
    clean_2 = rng.normal(0.0, 0.1, size=(16, dim))
    fail_2 = clean_2 + 1.5 * np.eye(dim)[1]

    surgeon = WeightSpaceSurgeon()
    ranked = surgeon.rank_layers(
        {
            "layers.11.mlp.down_proj": (fail_1, clean_1),
            "layers.15.mlp.down_proj": (fail_2, clean_2),
        },
        rank=1,
    )
    assert ranked[0].layer_name == "layers.11.mlp.down_proj"

    weights = {
        "layers.11.mlp.down_proj": rng.normal(0.0, 0.5, size=(dim, 40)),
        "layers.15.mlp.down_proj": rng.normal(0.0, 0.5, size=(dim, 40)),
    }
    patches = surgeon.synthesize_multilayer_patches(
        ranked, weights, top_layers=2, alpha=0.5, column_fraction=0.25
    )
    assert len(patches) == 2
    for patch in patches:
        # 25% of 40 columns = 10 active columns, 30 zeroed columns.
        col_norms = np.linalg.norm(patch.adapter.a, axis=0)
        assert int(np.sum(col_norms > 0.0)) == 10
        assert int(np.sum(col_norms == 0.0)) == 30


def test_inspect_reports_failed_integrity_and_exits_nonzero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    units_dir = tmp_path / "units"
    reg = SkillUnitRegistry(units_dir)
    reg.propose(
        unit_id="u-inspect",
        version="1.0.0",
        title="Inspect",
        summary="Test",
        proposer="tools",
        mode=AdaptationMode.DYNAMIC_EXPANSION_FP32,
        deltas=[build_dummy_delta("layers.18.mlp.down_proj", rank=2)],
        base_kernel_id="mock/base",
    )

    assert cli_main(["inspect", str(units_dir)]) == 0
    ok_out = json.loads(capsys.readouterr().out)
    assert ok_out[0]["integrity"] == "verified"

    # Tamper payload -> inspect must NOT crash and must report integrity="failed" + exit 1.
    (units_dir / "u-inspect" / "payload.scun").write_bytes(b"corrupt")
    assert cli_main(["inspect", str(units_dir)]) == 1
    bad_out = json.loads(capsys.readouterr().out)
    assert bad_out[0]["integrity"] == "failed"
    assert "SHA-256 mismatch" in bad_out[0]["error"]


def test_proving_ground_blocks_accidental_rerun_without_corrupting_telemetry(
    tmp_path: Path,
) -> None:
    work = tmp_path / "demo"
    run_proving_ground(work, seed=42)
    telem_path = work / "telemetry" / "trajectories.jsonl"
    orig_lines = len(telem_path.read_text(encoding="utf-8").splitlines())

    # Second run without overwrite=True must fail BEFORE appending duplicate telemetry.
    with pytest.raises(GovernanceError, match="already contains Skill Units"):
        run_proving_ground(work, seed=42)
    assert len(telem_path.read_text(encoding="utf-8").splitlines()) == orig_lines

    # Explicit overwrite=True succeeds cleanly.
    run_proving_ground(work, seed=42, overwrite=True)
    assert len(telem_path.read_text(encoding="utf-8").splitlines()) == orig_lines


def test_mlx_export_uses_detected_layer_prefix_and_last_n_num_layers(tmp_path: Path) -> None:
    mx = pytest.importorskip("mlx.core")
    from subcor.backends.mlx_kernel import export_mlx_lora_safetensors

    reg = SkillUnitRegistry(tmp_path / "units")
    reg.propose(
        unit_id="u-mlx-exp",
        version="1.0.0",
        title="MLX Export",
        summary="Test",
        proposer="tools",
        mode=AdaptationMode.DYNAMIC_EXPANSION_FP32,
        deltas=[
            build_dummy_delta("layers.18.mlp.down_proj", d_out=16, d_in=32, rank=2),
            build_dummy_delta("layers.23.mlp.down_proj", d_out=16, d_in=32, rank=2),
        ],
        base_kernel_id="mlx-community/Qwen3.5-0.8B-4bit",
    )
    out = export_mlx_lora_safetensors(
        "u-mlx-exp",
        reg,
        tmp_path / "st",
        layer_prefix="language_model.model.",
        total_layers=24,
    )
    cfg = json.loads((out / "adapter_config.json").read_text(encoding="utf-8"))
    assert cfg["num_layers"] == 6  # 24 - 18 = last 6 layers
    assert cfg["lora_parameters"]["keys"] == ["mlp.down_proj"]
    weights = mx.load(str(out / "adapters.safetensors"))
    assert "language_model.model.layers.18.mlp.down_proj.lora_a" in weights
    assert "language_model.model.layers.23.mlp.down_proj.lora_b" in weights
