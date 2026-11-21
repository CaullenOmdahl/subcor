"""End-to-end Micro-Intern Proving Ground for SLAINGI.

Constructs a real multi-layer residual iterative reasoning network
(MicroInternNet) afflicted with two distinct failure modes mirroring the
empirical walls identified in weight-space programming research:

  1. Structural Looping Defect (Task Family A-Loop):
     The frozen base network already knows how to solve Family A, but a
     trigger feature activates a self-reinforcing loop direction in layer_1
     that suppresses halting until max_steps is exhausted.
     -> Fixed by zero-gradient Contrastive Weight-Space Surgery.

  2. Missing Skill Circuit (Task Family B):
     The frozen base network lacks a circuit for Family B (~chance accuracy).
     Weight-space ablation cannot invent this capability.
     -> Fixed by function-preserving dynamic low-rank expansion + Binary STE
        adaptation on verified senior corrections, compressed to < 1 bit/weight.

Both adaptations flow through the governed SkillUnitRegistry (SHA-256 integrity,
locked Conformance Gate, no self-approval, atomic rollback, and CaPheOS export).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from slaingi.expansion import ElasticLinear, LowRankAdapter
from slaingi.surgeon import WeightSpaceSurgeon
from slaingi.telemetry import TrajectoryStore
from slaingi.types import (
    AdaptationMode,
    DeltaEncoding,
    GovernanceError,
    SkillUnitManifest,
    TrajectoryRecord,
    VerificationEvidence,
)
from slaingi.units import SkillUnitRegistry, build_layer_delta


@dataclass(frozen=True)
class TaskBatch:
    """Labeled inputs for one task slice."""

    family: str
    inputs: np.ndarray  # (n, dim)
    labels: np.ndarray  # (n,)


@dataclass(frozen=True)
class EvalMetrics:
    """Observable evaluation metrics on a TaskBatch."""

    family: str
    accuracy: float
    step_limit_rate: float
    mean_steps: float


@dataclass(frozen=True)
class ProvingGroundReport:
    """Complete verified report of one end-to-end SLAINGI proving-ground run."""

    baseline_clean: EvalMetrics
    baseline_loop: EvalMetrics
    baseline_novel: EvalMetrics

    after_surgery_clean: EvalMetrics
    after_surgery_loop: EvalMetrics
    after_surgery_novel: EvalMetrics

    after_expansion_clean: EvalMetrics
    after_expansion_loop: EvalMetrics
    after_expansion_novel: EvalMetrics

    after_rollback_clean: EvalMetrics
    after_rollback_loop: EvalMetrics
    after_rollback_novel: EvalMetrics

    detected_loop_layer: str
    zero_perturbation_max_diff: float
    unverified_filtered_count: int
    self_approval_blocked: bool
    poisoned_patch_rejected: bool

    surgery_unit: SkillUnitManifest
    skill_unit: SkillUnitManifest
    exported_capheos_paths: tuple[Path, ...]


class MicroInternNet:
    """2-layer iterative residual reasoning network over strictly frozen base weights."""

    def __init__(
        self, *, dim: int = 64, num_classes: int = 4, max_steps: int = 12, seed: int = 42
    ) -> None:
        if dim < 32:
            raise ValueError("dim must be >= 32")
        self.dim = dim
        self.num_classes = num_classes
        self.max_steps = max_steps

        rng = np.random.default_rng(seed)
        q, _ = np.linalg.qr(rng.normal(size=(dim, dim)))
        self._basis = q

        # Dedicated orthogonal subspaces in R^dim:
        self.class_proto = q[:, 0:num_classes]  # (dim, 4)
        self.q_halt = q[:, 4]
        self.q_loop = q[:, 5]
        self.q_trig_hidden = q[:, 6]

        self.in_family_a = q[:, 8 : 8 + num_classes]  # (dim, 4)
        self.in_halt_cue = q[:, 12]
        self.in_loop_trigger = q[:, 13]
        self.in_family_b = q[:, 16 : 16 + num_classes]  # (dim, 4)

        # Build frozen base weights for layer_0 (input encoder):
        # Encodes Family A -> class_proto, halt cue -> q_halt, trigger -> q_trig_hidden.
        # Deliberately has NO circuit for Family B (maps Family B subspace to ~0).
        w0 = (
            1.6 * (self.class_proto @ self.in_family_a.T)
            + 1.6 * np.outer(self.q_halt, self.in_halt_cue)
            + 1.6 * np.outer(self.q_trig_hidden, self.in_loop_trigger)
            + rng.normal(0.0, 0.015, size=(dim, dim))
        )

        # Build frozen base weights for layer_1 (iterative reasoning refinement):
        # Healthy refinement on class_proto and q_halt, PLUS an injected structural
        # looping defect that fires q_loop from q_trig_hidden and self-reinforces q_loop.
        w1 = (
            0.35 * (self.class_proto @ self.class_proto.T)
            + 0.45 * np.outer(self.q_halt, self.q_halt)
            + 2.6 * np.outer(self.q_loop, self.q_trig_hidden)
            + 2.4 * np.outer(self.q_loop, self.q_loop)
            + rng.normal(0.0, 0.01, size=(dim, dim))
        )

        # Frozen readout head & halt head.
        w_readout = 3.0 * self.class_proto.T  # (num_classes, dim)
        w_halt = 3.5 * self.q_halt - 6.5 * self.q_loop

        self.layers: dict[str, ElasticLinear] = {
            "layer_0": ElasticLinear(w0),
            "layer_1": ElasticLinear(w1),
            "readout": ElasticLinear(w_readout),
        }
        w_halt_ro = w_halt.copy()
        w_halt_ro.setflags(write=False)
        self._w_halt = w_halt_ro
        self._halt_threshold = 0.65

    def mount_unit(self, unit_id: str, registry: SkillUnitRegistry) -> None:
        """Verify SHA-256 integrity and hot-swap a Skill Unit onto the frozen layers."""
        deltas = registry.load_verified_deltas(unit_id)
        for delta in deltas:
            adapter = delta.to_adapter(adapter_name=f"unit:{unit_id}:{delta.layer_name}")
            self.layers[delta.layer_name].mount_adapter(adapter)

    def unmount_unit(self, unit_id: str) -> None:
        """Atomically remove all adapters belonging to unit_id (instant rollback)."""
        prefix = f"unit:{unit_id}:"
        for layer in self.layers.values():
            for name in list(layer.adapter_names):
                if name.startswith(prefix):
                    layer.unmount_adapter(name)

    def run_single(self, x: np.ndarray) -> tuple[int, int, bool, dict[str, np.ndarray]]:
        """Execute one iterative reasoning pass, capturing per-layer activation traces."""
        x_row = np.asarray(x, dtype=np.float64).reshape(1, -1)
        h0 = np.tanh(self.layers["layer_0"].forward(x_row))[0]

        h = h0
        layer1_sum = np.zeros(self.dim, dtype=np.float64)
        steps_taken = 0
        halted = False

        for step in range(1, self.max_steps + 1):
            steps_taken = step
            u1 = self.layers["layer_1"].forward(h.reshape(1, -1))[0]
            layer1_sum += u1
            h = np.tanh(h + 0.5 * u1)
            halt_logit = float(np.dot(self._w_halt, h))
            halt_prob = 1.0 / (1.0 + float(np.exp(-np.clip(halt_logit, -20.0, 20.0))))
            if halt_prob >= self._halt_threshold:
                halted = True
                break

        logits = self.layers["readout"].forward(h.reshape(1, -1))[0]
        pred = int(np.argmax(logits))
        activations = {
            "layer_0": h0,
            "layer_1": layer1_sum / steps_taken,
        }
        return pred, steps_taken, (not halted), activations

    def evaluate(self, batch: TaskBatch) -> EvalMetrics:
        """Evaluate accuracy (counting step-limit exhaustion as failure) and loop rate."""
        n = batch.inputs.shape[0]
        correct = 0
        loop_hits = 0
        total_steps = 0

        for i in range(n):
            pred, steps, hit_limit, _ = self.run_single(batch.inputs[i])
            total_steps += steps
            if hit_limit:
                loop_hits += 1
            elif pred == int(batch.labels[i]):
                correct += 1

        return EvalMetrics(
            family=batch.family,
            accuracy=correct / n,
            step_limit_rate=loop_hits / n,
            mean_steps=total_steps / n,
        )

    def sample_tasks(
        self,
        *,
        n_per_family: int = 80,
        seed: int = 100,
    ) -> tuple[TaskBatch, TaskBatch, TaskBatch]:
        """Generate deterministic (family_a_clean, family_a_loop, family_b_novel) batches."""
        rng = np.random.default_rng(seed)

        def make_batch(
            family: str,
            input_basis: np.ndarray,
            *,
            with_loop_trigger: bool,
        ) -> TaskBatch:
            labels = rng.integers(0, self.num_classes, size=n_per_family)
            xs = np.zeros((n_per_family, self.dim), dtype=np.float64)
            for i, lbl in enumerate(labels):
                vec = (
                    1.2 * input_basis[:, int(lbl)]
                    + 1.0 * self.in_halt_cue
                    + rng.normal(0.0, 0.12, size=self.dim)
                )
                if with_loop_trigger:
                    vec = vec + 1.3 * self.in_loop_trigger
                xs[i] = vec
            return TaskBatch(family=family, inputs=xs, labels=labels)

        return (
            make_batch("family_a_clean", self.in_family_a, with_loop_trigger=False),
            make_batch("family_a_loop", self.in_family_a, with_loop_trigger=True),
            make_batch("family_b_novel", self.in_family_b, with_loop_trigger=False),
        )


def run_proving_ground(
    work_dir: Path,
    *,
    export_capheos_dir: Path | None = None,
    seed: int = 42,
) -> ProvingGroundReport:
    """Execute the complete SLAINGI proving-ground experiment end-to-end."""
    work_dir.mkdir(parents=True, exist_ok=True)
    store = TrajectoryStore(work_dir / "telemetry" / "trajectories.jsonl")
    registry = SkillUnitRegistry(work_dir / "units")

    net = MicroInternNet(dim=64, num_classes=4, max_steps=12, seed=seed)

    # Separate training/reflection split and locked held-out conformance split.
    train_clean, train_loop, train_novel = net.sample_tasks(n_per_family=64, seed=seed + 1)
    eval_clean, eval_loop, eval_novel = net.sample_tasks(n_per_family=80, seed=seed + 2)

    # ------------------------------------------------------------------
    # Phase 0: Baseline evaluation & verified Intern telemetry capture
    # ------------------------------------------------------------------
    baseline_clean = net.evaluate(eval_clean)
    baseline_loop = net.evaluate(eval_loop)
    baseline_novel = net.evaluate(eval_novel)

    records: list[TrajectoryRecord] = []
    for batch in (train_clean, train_loop, train_novel):
        for i in range(batch.inputs.shape[0]):
            pred, steps, hit_limit, acts = net.run_single(batch.inputs[i])
            expected = int(batch.labels[i])
            succeeded = (not hit_limit) and (pred == expected)
            evidence = VerificationEvidence(
                verified=True,
                verifier="deterministic_task_verifier"
                if batch.family != "family_b_novel"
                else "senior_seat_ceo",
                method="deterministic_verifier"
                if batch.family != "family_b_novel"
                else "senior_seat_review",
            )
            records.append(
                TrajectoryRecord(
                    trajectory_id=f"{batch.family}-{i:03d}",
                    task_family=batch.family,
                    seat="intern",
                    input_vector=batch.inputs[i].tolist(),
                    predicted_label=pred,
                    expected_label=expected,
                    steps_taken=steps,
                    hit_step_limit=hit_limit,
                    succeeded=succeeded,
                    evidence=evidence,
                    layer_activations={k: v.tolist() for k, v in acts.items()},
                    senior_correction_label=expected if not succeeded else None,
                )
            )

    # Inject deliberately UNVERIFIED poisoned trajectories to prove the Verified
    # Evidence Gate strips them out before diagnosis or learning.
    rng_poison = np.random.default_rng(seed + 99)
    unverified_count = 12
    for j in range(unverified_count):
        fake_vec = rng_poison.normal(0.0, 5.0, size=net.dim).tolist()
        records.append(
            TrajectoryRecord(
                trajectory_id=f"unverified-noise-{j:03d}",
                task_family="family_b_novel",
                seat="intern",
                input_vector=fake_vec,
                predicted_label=0,
                expected_label=(j % net.num_classes),
                steps_taken=1,
                hit_step_limit=False,
                succeeded=False,
                evidence=VerificationEvidence(
                    verified=False,
                    verifier="unconfirmed",
                    method="unverified_draft",
                    notes="Must be excluded by Verified Evidence Gate",
                ),
                layer_activations={
                    "layer_0": fake_vec,
                    "layer_1": fake_vec,
                },
                senior_correction_label=(j % net.num_classes),
            )
        )

    store.append_many(records)
    total_logged = len(store.load_all())
    total_verified = len(store.load_verified())
    unverified_filtered = total_logged - total_verified

    # ------------------------------------------------------------------
    # Phase 1: Zero-gradient Contrastive Weight-Space Surgery
    # ------------------------------------------------------------------
    surgeon = WeightSpaceSurgeon()
    layer_contrast: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for layer_name in ("layer_0", "layer_1"):
        fail_act, clean_act = store.contrastive_activations(
            layer_name,
            require_step_limit_for_failures=True,
        )
        layer_contrast[layer_name] = (fail_act, clean_act)

    ranked = surgeon.rank_layers(layer_contrast, rank=1)
    top_diag = ranked[0]
    patch = surgeon.synthesize_ablation_patch(
        top_diag,
        net.layers[top_diag.layer_name].w_base,
        adapter_name="proposed-loop-ablation",
        alpha=1.0,
    )

    surgery_delta = build_layer_delta(
        patch.layer_name,
        patch.adapter,
        encoding=DeltaEncoding.SUB1BIT_BINARY,
        sub1bit_rank=1,
    )
    surgery_unit = registry.propose(
        unit_id="loop-ablation-v1",
        version="1.0.0",
        title="Structural Loop Ablation Patch",
        summary=(
            f"Zero-gradient rank-1 contrastive ablation targeting {patch.layer_name} "
            f"to eliminate thinking-loop step-limit failures."
        ),
        proposer="tools",
        mode=AdaptationMode.WEIGHT_SPACE_SURGERY,
        deltas=[surgery_delta],
    )

    # Evaluate candidate unit inside sandbox mount before conformance certification.
    net.mount_unit(surgery_unit.unit_id, registry)
    cand_clean = net.evaluate(eval_clean)
    cand_loop = net.evaluate(eval_loop)
    net.unmount_unit(surgery_unit.unit_id)

    surgery_unit = registry.record_conformance(
        surgery_unit.unit_id,
        baseline_before=baseline_clean.accuracy,
        baseline_after=cand_clean.accuracy,
        target_before=baseline_loop.accuracy,
        target_after=cand_loop.accuracy,
        max_regression_tolerance=0.02,
        min_target_improvement=0.25,
    )

    # Prove self-approval is strictly forbidden.
    self_approval_blocked = False
    try:
        registry.promote(surgery_unit.unit_id, approver="tools")
    except GovernanceError:
        self_approval_blocked = True

    # Independent risk seat approves and mounts the verified unit.
    surgery_unit = registry.promote(surgery_unit.unit_id, approver="risk")
    net.mount_unit(surgery_unit.unit_id, registry)

    after_surgery_clean = net.evaluate(eval_clean)
    after_surgery_loop = net.evaluate(eval_loop)
    after_surgery_novel = net.evaluate(eval_novel)

    # ------------------------------------------------------------------
    # Phase 2: Function-Preserving Dynamic Expansion + Binary STE + Sub-1-Bit
    # ------------------------------------------------------------------
    x_b, y_b = store.verified_correction_dataset(task_family="family_b_novel")
    x_a, y_a = store.verified_correction_dataset(task_family="family_a_clean")

    # Build verified target hidden representations in layer_0 output space:
    # Family B inputs map to their verified class prototype + clean halt cue;
    # Family A replay anchors existing base outputs so baseline stays intact.
    y_target_b = 1.6 * net.class_proto[:, y_b].T + 1.6 * net.q_halt[np.newaxis, :]
    y_target_a = net.layers["layer_0"].forward(x_a)

    x_train = np.vstack([x_b, x_a])
    y_train = np.vstack([y_target_b, y_target_a])

    # Prove strict zero-perturbation invariant on dynamic rank expansion.
    probe_x = eval_clean.inputs[:16]
    before_expand = net.layers["layer_0"].forward(probe_x)
    skill_adapter = net.layers["layer_0"].expand(
        name="candidate-novel-skill",
        rank=4,
        binary_ste=True,
        seed=seed + 7,
    )
    after_expand = net.layers["layer_0"].forward(probe_x)
    zero_perturbation_max_diff = float(np.max(np.abs(after_expand - before_expand)))

    # Train ONLY the newly expanded binary STE adapter (base weights stay frozen).
    net.layers["layer_0"].fit_adapter_mse(
        "candidate-novel-skill",
        x_train,
        y_train,
        steps=280,
        lr=0.04,
        l2_reg=1e-4,
    )

    # Compress learned adapter into bit-packed sub-1-bit representation & unmount draft.
    skill_delta = build_layer_delta(
        "layer_0",
        skill_adapter,
        encoding=DeltaEncoding.SUB1BIT_BINARY,
        sub1bit_rank=4,
    )
    net.layers["layer_0"].unmount_adapter("candidate-novel-skill")

    skill_unit = registry.propose(
        unit_id="novel-skill-v1",
        version="1.0.0",
        title="Family-B Routing Skill Adapter (Sub-1-Bit)",
        summary=(
            "Dynamically expanded rank-4 Binary-STE adapter trained on verified senior "
            "corrections and packed to < 1 bit/weight."
        ),
        proposer="tools",
        mode=AdaptationMode.DYNAMIC_EXPANSION_STE,
        deltas=[skill_delta],
    )

    # Sandbox evaluation against locked conformance suite (baseline = both Family A slices).
    baseline_before_p2 = 0.5 * (after_surgery_clean.accuracy + after_surgery_loop.accuracy)
    net.mount_unit(skill_unit.unit_id, registry)
    cand2_clean = net.evaluate(eval_clean)
    cand2_loop = net.evaluate(eval_loop)
    cand2_novel = net.evaluate(eval_novel)
    net.unmount_unit(skill_unit.unit_id)

    baseline_after_p2 = 0.5 * (cand2_clean.accuracy + cand2_loop.accuracy)
    skill_unit = registry.record_conformance(
        skill_unit.unit_id,
        baseline_before=baseline_before_p2,
        baseline_after=baseline_after_p2,
        target_before=after_surgery_novel.accuracy,
        target_after=cand2_novel.accuracy,
        max_regression_tolerance=0.02,
        min_target_improvement=0.30,
    )
    skill_unit = registry.promote(skill_unit.unit_id, approver="risk")
    net.mount_unit(skill_unit.unit_id, registry)

    after_expansion_clean = net.evaluate(eval_clean)
    after_expansion_loop = net.evaluate(eval_loop)
    after_expansion_novel = net.evaluate(eval_novel)

    # ------------------------------------------------------------------
    # Phase 3: Governance Proof — Reject Poisoned Patch + Atomic Rollback
    # ------------------------------------------------------------------
    rng_bad = np.random.default_rng(seed + 123)
    poison_adapter = LowRankAdapter(
        name="poison-draft",
        a=rng_bad.normal(0.0, 2.0, size=(2, net.dim)),
        b=rng_bad.normal(0.0, 2.0, size=(net.dim, 2)),
        scale=1.0,
    )
    poison_delta = build_layer_delta(
        "layer_0",
        poison_adapter,
        encoding=DeltaEncoding.SUB1BIT_BINARY,
        sub1bit_rank=2,
    )
    poison_unit = registry.propose(
        unit_id="poisoned-patch-v1",
        version="0.0.1",
        title="Deliberately Regressive Patch",
        summary="Negative test proving locked conformance gate blocks regressive patches.",
        proposer="tools",
        mode=AdaptationMode.MANUAL_TEST,
        deltas=[poison_delta],
    )
    net.mount_unit(poison_unit.unit_id, registry)
    poison_clean = net.evaluate(eval_clean)
    poison_novel = net.evaluate(eval_novel)
    net.unmount_unit(poison_unit.unit_id)

    poison_unit = registry.record_conformance(
        poison_unit.unit_id,
        baseline_before=after_expansion_clean.accuracy,
        baseline_after=poison_clean.accuracy,
        target_before=after_expansion_novel.accuracy,
        target_after=poison_novel.accuracy,
        max_regression_tolerance=0.02,
        min_target_improvement=0.05,
    )
    poisoned_patch_rejected = False
    if poison_unit.conformance is not None and not poison_unit.conformance.passed:
        try:
            registry.promote(poison_unit.unit_id, approver="risk")
        except GovernanceError:
            poisoned_patch_rejected = True

    # Export CaPheOS format-2 unit bundles before rollback while both units are ENABLED.
    exported_paths: list[Path] = []
    if export_capheos_dir is not None:
        exported_paths.append(
            registry.export_capheos_unit(surgery_unit.unit_id, export_capheos_dir)
        )
        exported_paths.append(registry.export_capheos_unit(skill_unit.unit_id, export_capheos_dir))

    # Demonstrate atomic rollback of novel-skill-v1 -> exact restoration to Phase 1.
    registry.rollback(
        skill_unit.unit_id, actor="risk", reason="Proving-ground rollback verification"
    )
    net.unmount_unit(skill_unit.unit_id)

    after_rollback_clean = net.evaluate(eval_clean)
    after_rollback_loop = net.evaluate(eval_loop)
    after_rollback_novel = net.evaluate(eval_novel)

    return ProvingGroundReport(
        baseline_clean=baseline_clean,
        baseline_loop=baseline_loop,
        baseline_novel=baseline_novel,
        after_surgery_clean=after_surgery_clean,
        after_surgery_loop=after_surgery_loop,
        after_surgery_novel=after_surgery_novel,
        after_expansion_clean=after_expansion_clean,
        after_expansion_loop=after_expansion_loop,
        after_expansion_novel=after_expansion_novel,
        after_rollback_clean=after_rollback_clean,
        after_rollback_loop=after_rollback_loop,
        after_rollback_novel=after_rollback_novel,
        detected_loop_layer=top_diag.layer_name,
        zero_perturbation_max_diff=zero_perturbation_max_diff,
        unverified_filtered_count=unverified_filtered,
        self_approval_blocked=self_approval_blocked,
        poisoned_patch_rejected=poisoned_patch_rejected,
        surgery_unit=surgery_unit,
        skill_unit=skill_unit,
        exported_capheos_paths=tuple(exported_paths),
    )
