"""Sub-1-bit packing of a Study Cycle must not hold the shared kernel lock.

On the llama.cpp backend, packing a 4B unit took ~210 s of CPU time; while it ran under
TenantManager.lock every tenant's chat waited. Kernels that expose
train_skill_adapters + pack_skill_deltas get training under the lock and packing outside it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from subcor.backends.llamacpp_kernel import TrainedAdapters
from subcor.expansion import LowRankAdapter
from subcor.server import TenantManager, build_dummy_delta
from test_p0_correctness_and_eval import _MinimalKernel


class _SplitTrainingKernel(_MinimalKernel):
    def __init__(self) -> None:
        self.manager: TenantManager | None = None
        self.lock_held_during_training: bool | None = None
        self.lock_held_during_packing: bool | None = None
        self.legacy_called = False

    def train_skill_adapters(self, **_: Any) -> TrainedAdapters:
        assert self.manager is not None
        self.lock_held_during_training = self.manager.lock.locked()
        adapter = LowRankAdapter.zero_initialized(self.tap_layer, 16, 16, 2)
        return TrainedAdapters(
            layer_names=(self.tap_layer,),
            adapters={self.tap_layer: adapter},
            initial_loss=1.5,
            final_loss=0.05,
            zero_perturbation_max_diff=0.0,
        )

    def pack_skill_deltas(self, trained: TrainedAdapters, *, sub1bit_rank: int) -> list:
        assert self.manager is not None
        self.lock_held_during_packing = self.manager.lock.locked()
        return [build_dummy_delta(self.tap_layer, rank=2)]

    def train_skill_deltas(self, **_: Any):  # pragma: no cover - must not be used
        self.legacy_called = True
        raise AssertionError("split-capable kernels must not use train_skill_deltas")


def test_study_packs_sub1bit_outside_kernel_lock(tmp_path: Path) -> None:
    kernel = _SplitTrainingKernel()
    mgr = TenantManager(state_dir=tmp_path / "st", kernel=kernel, initial_tenants=("t",))
    kernel.manager = mgr
    mgr.ingest_trajectories(
        "t",
        [
            {
                "task_family": "ops",
                "prompt": f"Event: Invoice #{i} over cap",
                "response": "bad",
                "senior_correction": '{"code":"FIN-CAP"}',
                "evidence": {"verified": True, "verifier": "senior"},
            }
            for i in range(4)
        ],
    )
    res = mgr.run_study_cycle(
        "t",
        unit_id="ops-split-v1",
        version="1.0.0",
        task_family="ops",
        title="Ops",
        summary="Packing outside the lock",
    )
    assert kernel.lock_held_during_training is True
    assert kernel.lock_held_during_packing is False
    assert kernel.legacy_called is False
    assert res["initial_loss"] == 1.5 and res["final_loss"] == 0.05
    assert res["manifest"]["state"] in ("conformant", "dormant")
