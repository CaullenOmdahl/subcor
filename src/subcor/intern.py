"""Text-level Intern telemetry store enforcing the Verified Evidence Gate.

Extends SUBCOR's append-only telemetry contract to real LLM text generations:
  - Captures prompt, Intern response, token count, step/token-limit exhaustion,
    n-gram repetition ratio (for detecting thinking loops), verified senior
    corrections, and per-layer hidden-state activation snapshots.
  - Enforces the Verified Evidence Gate: unverified records never enter
    contrastive weight-space diagnosis or on-policy QLoRA distillation.
"""

from __future__ import annotations

import json
import threading
import warnings
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from subcor.types import VerificationError, VerificationEvidence


def ngram_repetition_ratio(text: str, *, n: int = 4) -> float:
    """Measure fraction of repeated whitespace n-grams in text (0.0 = unique, ->1.0 = loop)."""
    tokens = text.split()
    if len(tokens) < n + 1:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    unique = len(set(ngrams))
    return 1.0 - (unique / len(ngrams))


@dataclass(frozen=True)
class TextTrajectory:
    """Single live LLM Intern execution trace stored in append-only JSONL."""

    trajectory_id: str
    task_family: str
    seat: str
    prompt: str
    response: str
    tokens_generated: int
    hit_token_limit: bool
    repetition_ratio: float
    passed_verifier: bool
    evidence: VerificationEvidence
    senior_correction: str | None = None
    layer_activations: dict[str, list[float]] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    @property
    def is_verified(self) -> bool:
        return self.evidence.verified

    def to_dict(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "task_family": self.task_family,
            "seat": self.seat,
            "prompt": self.prompt,
            "response": self.response,
            "tokens_generated": self.tokens_generated,
            "hit_token_limit": self.hit_token_limit,
            "repetition_ratio": self.repetition_ratio,
            "passed_verifier": self.passed_verifier,
            "evidence": self.evidence.to_dict(),
            "senior_correction": self.senior_correction,
            "layer_activations": self.layer_activations,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TextTrajectory:
        return cls(
            trajectory_id=str(data["trajectory_id"]),
            task_family=str(data["task_family"]),
            seat=str(data["seat"]),
            prompt=str(data["prompt"]),
            response=str(data["response"]),
            tokens_generated=int(data["tokens_generated"]),
            hit_token_limit=bool(data["hit_token_limit"]),
            repetition_ratio=float(data["repetition_ratio"]),
            passed_verifier=bool(data["passed_verifier"]),
            evidence=VerificationEvidence.from_dict(data["evidence"]),
            senior_correction=(
                None if data.get("senior_correction") is None else str(data["senior_correction"])
            ),
            layer_activations={
                str(k): [float(x) for x in v] for k, v in data.get("layer_activations", {}).items()
            },
            timestamp=str(data.get("timestamp", datetime.now(UTC).isoformat())),
        )


class TextTrajectoryStore:
    """Append-only JSONL store for live LLM Intern traces and senior corrections."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._io_lock = threading.Lock()
        self.corrupted_line_count: int = 0

    def append(self, record: TextTrajectory) -> None:
        line = json.dumps(record.to_dict(), separators=(",", ":"), sort_keys=True)
        with self._io_lock, self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    def append_many(self, records: Iterable[TextTrajectory]) -> int:
        lines = [
            json.dumps(rec.to_dict(), separators=(",", ":"), sort_keys=True) + "\n"
            for rec in records
        ]
        if not lines:
            return 0
        with self._io_lock, self.path.open("a", encoding="utf-8") as fh:
            fh.writelines(lines)
        return len(lines)

    def load_all(self) -> list[TextTrajectory]:
        if not self.path.exists():
            return []
        out: list[TextTrajectory] = []
        corrupted = 0
        with self._io_lock, self.path.open("r", encoding="utf-8") as fh:
            for lineno, raw in enumerate(fh, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    out.append(TextTrajectory.from_dict(json.loads(raw)))
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                    corrupted += 1
                    warnings.warn(
                        f"Skipping corrupted text trajectory line {lineno} in {self.path}: {exc}",
                        RuntimeWarning,
                        stacklevel=2,
                    )
        self.corrupted_line_count = corrupted
        return out

    def load_verified(self, *, task_family: str | None = None) -> list[TextTrajectory]:
        """Return only verified trajectories; unverified records are strictly dropped."""
        rows = [r for r in self.load_all() if r.is_verified]
        if task_family is not None:
            rows = [r for r in rows if r.task_family == task_family]
        return rows

    def verified_distillation_pairs(
        self,
        *,
        task_family: str | None = None,
        include_passing_as_replay: bool = False,
    ) -> list[tuple[str, str]]:
        """Extract verified (prompt, target_response) pairs for on-policy distillation.

        Uses senior_correction when provided; optionally includes verified passing
        trajectories as replay anchors to prevent catastrophic forgetting.
        """
        pairs: list[tuple[str, str]] = []
        for rec in self.load_verified(task_family=task_family):
            if rec.senior_correction is not None:
                pairs.append((rec.prompt, rec.senior_correction))
            elif include_passing_as_replay and rec.passed_verifier and not rec.hit_token_limit:
                pairs.append((rec.prompt, rec.response))
        if not pairs:
            raise VerificationError(
                f"No verified distillation pairs available for task_family={task_family!r}"
            )
        return pairs

    def contrastive_activations(
        self,
        layer_name: str,
        *,
        task_family: str | None = None,
        require_token_limit: bool = True,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return verified (failing_activations, clean_activations) arrays for a layer."""
        failing: list[list[float]] = []
        clean: list[list[float]] = []
        for rec in self.load_verified(task_family=task_family):
            vec = rec.layer_activations.get(layer_name)
            if vec is None:
                continue
            if rec.passed_verifier and not rec.hit_token_limit:
                clean.append(vec)
            elif rec.hit_token_limit or (not require_token_limit and not rec.passed_verifier):
                failing.append(vec)

        if not failing or not clean:
            raise VerificationError(
                f"Insufficient verified contrastive activations for '{layer_name}': "
                f"failing={len(failing)}, clean={len(clean)}"
            )
        return np.asarray(failing, dtype=np.float64), np.asarray(clean, dtype=np.float64)
