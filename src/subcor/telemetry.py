"""Append-only JSONL trajectory store enforcing the Verified Evidence Gate.

Core invariant ("Prevent Blind Leading the Blind"):
Only trajectories backed by verified evidence (deterministic verifier pass or
explicit senior/owner sign-off) may enter weight-space diagnosis or adapter
distillation datasets.
"""

from __future__ import annotations

import json
import threading
import warnings
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from subcor.types import TrajectoryRecord, VerificationError


class TrajectoryStore:
    """Append-only JSONL store for Intern trajectories and activation traces."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._io_lock = threading.Lock()
        self.corrupted_line_count: int = 0

    def append(self, record: TrajectoryRecord) -> None:
        """Append one immutable trajectory record to the JSONL log."""
        line = json.dumps(record.to_dict(), separators=(",", ":"), sort_keys=True)
        with self._io_lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def append_many(self, records: Iterable[TrajectoryRecord]) -> int:
        """Append multiple trajectory records and return the count written."""
        lines = [
            json.dumps(record.to_dict(), separators=(",", ":"), sort_keys=True) + "\n"
            for record in records
        ]
        if not lines:
            return 0
        with self._io_lock, self.path.open("a", encoding="utf-8") as fh:
            fh.writelines(lines)
        return len(lines)

    def load_all(self) -> list[TrajectoryRecord]:
        """Read all valid stored trajectories in append order, skipping corrupted lines."""
        if not self.path.exists():
            return []
        records: list[TrajectoryRecord] = []
        corrupted = 0
        with self._io_lock, self.path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    records.append(TrajectoryRecord.from_dict(json.loads(raw)))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    corrupted += 1
                    warnings.warn(
                        f"Skipping corrupted trajectory line {lineno} in {self.path}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        self.corrupted_line_count = corrupted
        return records

    def load_verified(self, task_family: str | None = None) -> list[TrajectoryRecord]:
        """Return only trajectories backed by verified evidence.

        Unverified trajectories are strictly excluded so hallucinated or
        unconfirmed agent outputs never poison downstream learning.
        """
        verified = [r for r in self.load_all() if r.is_verified]
        if task_family is not None:
            verified = [r for r in verified if r.task_family == task_family]
        return verified

    def contrastive_activations(
        self,
        layer_name: str,
        *,
        task_family: str | None = None,
        require_step_limit_for_failures: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract verified (failing_activations, clean_activations) matrices for a layer.

        Returns two 2D float64 arrays of shape (n_fail, dim) and (n_clean, dim).
        Raises VerificationError if verified failing or clean samples are absent.
        """
        verified = self.load_verified(task_family=task_family)
        failing: list[list[float]] = []
        clean: list[list[float]] = []

        for rec in verified:
            act = rec.layer_activations.get(layer_name)
            if act is None:
                continue
            if rec.succeeded and not rec.hit_step_limit:
                clean.append(act)
            elif (not require_step_limit_for_failures and not rec.succeeded) or rec.hit_step_limit:
                failing.append(act)

        if not failing or not clean:
            raise VerificationError(
                f"Insufficient verified contrastive activations for layer '{layer_name}': "
                f"failing={len(failing)}, clean={len(clean)}"
            )

        return np.asarray(failing, dtype=np.float64), np.asarray(clean, dtype=np.float64)

    def verified_correction_dataset(
        self,
        *,
        task_family: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Export verified (inputs, target_labels) pairs for on-policy distillation.

        Uses senior_correction_label when present, falling back to verified
        expected_label. Never emits unverified records.
        """
        verified = self.load_verified(task_family=task_family)
        xs: list[list[float]] = []
        ys: list[int] = []

        for rec in verified:
            target = (
                rec.senior_correction_label
                if rec.senior_correction_label is not None
                else rec.expected_label
            )
            if target is None:
                continue
            xs.append(rec.input_vector)
            ys.append(target)

        if not xs:
            raise VerificationError(
                f"No verified labeled trajectories available for task_family={task_family!r}"
            )

        return np.asarray(xs, dtype=np.float64), np.asarray(ys, dtype=np.int64)
