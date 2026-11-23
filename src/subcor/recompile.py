"""Portable Skill Unit recompiler (`subcor recompile`).

Treats compiled `.scun` weights as base-kernel-specific binaries and
`skill_spec.json` as portable workflow source code: when you upgrade the
frozen base model (the "impact drill"), `recompile_units_for_kernel()` reads
every eligible unit's portable `skill_spec.json`, trains fresh sub-1-bit
weights + frozen-base `RouterGate` on the new kernel, evaluates each unit
against its locked conformance suite, and outputs `CONFORMANT` (or `DORMANT`
on failure) units tagged with the new `base_kernel_id`.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from subcor.server import ConformancePolicy, ServingKernel, TenantManager
from subcor.types import SkillUnitManifest, UnitState
from subcor.units import SkillUnitRegistry


@dataclass(frozen=True)
class RecompiledUnitOutcome:
    """Outcome for a single Skill Unit recompiled onto a new base kernel."""

    unit_id: str
    source_kernel_id: str
    target_kernel_id: str
    status: str  # "conformant" | "failed_conformance" | "skipped_no_spec" | "rejected_policy"
    manifest: SkillUnitManifest | None = None
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "source_kernel_id": self.source_kernel_id,
            "target_kernel_id": self.target_kernel_id,
            "status": self.status,
            "reason": self.reason,
            "manifest": None if self.manifest is None else self.manifest.to_dict(),
        }


@dataclass(frozen=True)
class RecompileReport:
    """Summary of recompiling a library of Skill Units onto a new base kernel."""

    source_units_dir: Path
    dest_state_dir: Path
    target_tenant: str
    target_kernel_id: str
    outcomes: tuple[RecompiledUnitOutcome, ...]

    @property
    def conformant_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "conformant")

    @property
    def failed_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "failed_conformance")

    @property
    def rejected_policy_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "rejected_policy")

    @property
    def skipped_count(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "skipped_no_spec")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_units_dir": str(self.source_units_dir),
            "dest_state_dir": str(self.dest_state_dir),
            "target_tenant": self.target_tenant,
            "target_kernel_id": self.target_kernel_id,
            "conformant": self.conformant_count,
            "failed_conformance": self.failed_count,
            "skipped_no_spec": self.skipped_count,
            "rejected_policy": self.rejected_policy_count,
            "outcomes": [o.to_dict() for o in self.outcomes],
        }


def recompile_units_for_kernel(
    *,
    source_units_dir: Path,
    dest_state_dir: Path,
    target_tenant: str,
    kernel: ServingKernel,
    include_states: tuple[UnitState, ...] = (UnitState.ENABLED, UnitState.CONFORMANT),
    steps_override: int | None = None,
    seed: int = 42,
    conformance_policy: ConformancePolicy | None = None,
) -> RecompileReport:
    """Recompile all portable Skill Units from `source_units_dir` onto `kernel`.

    Every recompiled unit passes through the locked Conformance Gate and lands in
    `CONFORMANT` (never self-enabled), preserving separation of duties. Thresholds stored
    in a spec are clamped to `conformance_policy` (the owner's limits): a spec can make
    the gate stricter, never looser.
    """
    policy = conformance_policy or ConformancePolicy()
    src_registry = SkillUnitRegistry(source_units_dir)
    dest_manager = TenantManager(
        state_dir=dest_state_dir,
        kernel=kernel,
        initial_tenants=(target_tenant,),
        conformance_policy=policy,
    )

    outcomes: list[RecompiledUnitOutcome] = []
    for idx, src_manifest in enumerate(src_registry.list_units()):
        if src_manifest.state not in include_states:
            continue

        spec = src_registry.load_skill_spec(src_manifest.unit_id)
        if spec is None:
            outcomes.append(
                RecompiledUnitOutcome(
                    unit_id=src_manifest.unit_id,
                    source_kernel_id=src_manifest.base_kernel_id,
                    target_kernel_id=kernel.model_id,
                    status="skipped_no_spec",
                    reason="No portable skill_spec.json stored in source Skill Unit",
                )
            )
            continue

        # Seed destination tenant's verified telemetry from the sanitized SkillSpec.
        if spec.training_pairs:
            dest_manager.ingest_trajectories(
                target_tenant,
                [
                    {
                        "task_family": spec.task_family,
                        "prompt": prompt,
                        "response": "",
                        "senior_correction": target,
                        "evidence": {
                            "verified": True,
                            "verifier": "portable_skill_spec",
                            "method": "skill_spec_recompile",
                        },
                    }
                    for prompt, target in spec.training_pairs
                ],
            )

        # Stored thresholds are clamped to the owner policy (stricter only). A spec whose
        # stored thresholds are invalid (e.g. NaN) is reported, not fatal to the batch;
        # any other error from the study cycle still propagates.
        try:
            tol, imp = policy.resolve(
                min(spec.max_regression_tolerance, policy.max_regression_tolerance),
                max(spec.min_target_improvement, policy.min_target_improvement),
            )
        except ValueError as exc:
            outcomes.append(
                RecompiledUnitOutcome(
                    unit_id=spec.unit_id,
                    source_kernel_id=src_manifest.base_kernel_id,
                    target_kernel_id=kernel.model_id,
                    status="rejected_policy",
                    reason=str(exc),
                )
            )
            continue
        res = dest_manager.run_study_cycle(
            target_tenant,
            unit_id=spec.unit_id,
            version=spec.version,
            task_family=spec.task_family,
            title=spec.title,
            summary=spec.summary,
            proposer=spec.proposer,
            system_prompt=spec.system_prompt,
            training_trajectories=(
                list(spec.training_trajectories) if spec.training_trajectories else None
            ),
            eval_cases=list(spec.eval_cases) if spec.eval_cases else None,
            baseline_cases=list(spec.baseline_cases) if spec.baseline_cases else None,
            negative_prompts=list(spec.negative_prompts) if spec.negative_prompts else None,
            rank=spec.rank,
            sub1bit_rank=spec.sub1bit_rank,
            steps=steps_override if steps_override is not None else spec.steps,
            lr=spec.lr,
            max_regression_tolerance=tol,
            min_target_improvement=imp,
            seed=seed + idx * 17,
        )
        new_manifest = SkillUnitManifest.from_dict(res["manifest"])
        passed = new_manifest.state is UnitState.CONFORMANT
        reason = (
            new_manifest.conformance.reason
            if new_manifest.conformance is not None
            else ("Passed" if passed else "Failed conformance")
        )
        outcomes.append(
            RecompiledUnitOutcome(
                unit_id=spec.unit_id,
                source_kernel_id=src_manifest.base_kernel_id,
                target_kernel_id=kernel.model_id,
                status="conformant" if passed else "failed_conformance",
                manifest=new_manifest,
                reason=reason,
            )
        )

    return RecompileReport(
        source_units_dir=source_units_dir,
        dest_state_dir=dest_state_dir,
        target_tenant=target_tenant,
        target_kernel_id=kernel.model_id,
        outcomes=tuple(outcomes),
    )
