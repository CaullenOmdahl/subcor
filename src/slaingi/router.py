"""Activation-space MoE Skill Router for SLAINGI ("Shared Dense Core + Routed Units").

Instead of stacking dense adapters unconditionally (which causes cross-skill
interference and general-knowledge regression on small models), each Skill Unit
ships with a compact, analytically fitted RouterGate:

  - Computed from frozen-base hidden activations of verified positive prompts
    (E+) vs. verified negative / general-baseline prompts (E-).
  - Zero gradient descent and immune to small-batch router collapse.
  - Strictly invariant to other installed Skill Units because prompt embeddings
    are tapped from the immutable shared base kernel.
  - When no expert's score clears its calibrated threshold, routing falls back
    to 0 active experts (100% pure frozen base kernel -> 0.0% regression).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _l2_normalize(vec: np.ndarray, *, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > eps else np.zeros_like(arr)
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return np.where(norms > eps, arr / np.maximum(norms, eps), 0.0)


@dataclass(frozen=True)
class RouterGate:
    """Compact contrastive centroid gate for a single routed Skill Unit."""

    unit_id: str
    tap_layer: str
    pos_centroid: np.ndarray  # unit vector (dim,)
    neg_centroid: np.ndarray  # unit vector (dim,)
    threshold: float
    pos_mean_score: float
    neg_mean_score: float

    def __post_init__(self) -> None:
        if self.pos_centroid.ndim != 1 or self.neg_centroid.ndim != 1:
            raise ValueError("RouterGate centroids must be 1D vectors")
        if self.pos_centroid.shape != self.neg_centroid.shape:
            raise ValueError("pos_centroid and neg_centroid must share the same dimension")

    @property
    def dim(self) -> int:
        return int(self.pos_centroid.shape[0])

    @property
    def margin(self) -> float:
        return self.pos_mean_score - self.neg_mean_score

    def score(self, embedding: np.ndarray) -> float:
        """Return contrastive cosine margin cos(e, mu+) - cos(e, mu-) in [-2, 2]."""
        e = _l2_normalize(embedding)
        if e.shape != self.pos_centroid.shape:
            raise ValueError(
                f"Embedding dim {e.shape} does not match gate '{self.unit_id}' dim "
                f"{self.pos_centroid.shape}"
            )
        return float(np.dot(e, self.pos_centroid) - np.dot(e, self.neg_centroid))

    def matches(self, embedding: np.ndarray) -> bool:
        return self.score(embedding) >= self.threshold

    @classmethod
    def fit(
        cls,
        *,
        unit_id: str,
        tap_layer: str,
        pos_embeddings: np.ndarray,
        neg_embeddings: np.ndarray,
        safety_bias: float = 0.0,
    ) -> RouterGate:
        """Fit a contrastive centroid gate analytically from verified embeddings.

        `safety_bias` > 0 raises the threshold toward positives (more conservative,
        reducing false-positive activation on general prompts).
        """
        pos = _l2_normalize(np.asarray(pos_embeddings, dtype=np.float64))
        neg = _l2_normalize(np.asarray(neg_embeddings, dtype=np.float64))
        if pos.ndim != 2 or neg.ndim != 2 or pos.shape[0] == 0 or neg.shape[0] == 0:
            raise ValueError("pos_embeddings and neg_embeddings must be non-empty 2D arrays")
        if pos.shape[1] != neg.shape[1]:
            raise ValueError("Embedding dimension mismatch between positive and negative sets")

        mu_pos = _l2_normalize(np.mean(pos, axis=0))
        mu_neg = _l2_normalize(np.mean(neg, axis=0))

        pos_scores = (pos @ mu_pos) - (pos @ mu_neg)
        neg_scores = (neg @ mu_pos) - (neg @ mu_neg)

        # Calibrate threshold conservatively above observed negatives to prevent
        # false-positive routing on unrelated general/instruction prompts.
        pos_low = float(np.quantile(pos_scores, 0.15))
        neg_high = float(np.quantile(neg_scores, 0.90))
        neg_max = float(np.max(neg_scores))
        midpoint = 0.5 * (pos_low + neg_high) + safety_bias
        # Push threshold toward positives (65/35 blend between neg_max and pos_low) when separable.
        if pos_low > neg_max:
            conservative = neg_max + 0.35 * (pos_low - neg_max) + safety_bias
            threshold = max(midpoint, conservative)
        else:
            threshold = max(midpoint, float(np.mean(neg_scores)) + 1e-4)

        return cls(
            unit_id=unit_id,
            tap_layer=tap_layer,
            pos_centroid=mu_pos,
            neg_centroid=mu_neg,
            threshold=float(threshold),
            pos_mean_score=float(np.mean(pos_scores)),
            neg_mean_score=float(np.mean(neg_scores)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "unit_id": self.unit_id,
            "tap_layer": self.tap_layer,
            "pos_centroid": [round(float(x), 6) for x in self.pos_centroid],
            "neg_centroid": [round(float(x), 6) for x in self.neg_centroid],
            "threshold": round(self.threshold, 6),
            "pos_mean_score": round(self.pos_mean_score, 6),
            "neg_mean_score": round(self.neg_mean_score, 6),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RouterGate:
        return cls(
            unit_id=str(data["unit_id"]),
            tap_layer=str(data["tap_layer"]),
            pos_centroid=_l2_normalize(np.asarray(data["pos_centroid"], dtype=np.float64)),
            neg_centroid=_l2_normalize(np.asarray(data["neg_centroid"], dtype=np.float64)),
            threshold=float(data["threshold"]),
            pos_mean_score=float(data["pos_mean_score"]),
            neg_mean_score=float(data["neg_mean_score"]),
        )


@dataclass(frozen=True)
class RoutingDecision:
    """Observable routing outcome for a single prompt."""

    selected_units: tuple[str, ...]
    scores: dict[str, float]
    thresholds: dict[str, float]

    @property
    def fell_back_to_base(self) -> bool:
        return len(self.selected_units) == 0


class MoESkillRouter:
    """Top-k Skill Unit router over a frozen shared dense core.

    Selects up to `top_k` registered Skill Units whose contrastive score clears
    their calibrated threshold (ranked by margin above threshold). If no unit
    clears its threshold, selects 0 units so inference runs on the pure frozen
    shared base kernel with exact zero interference.
    """

    def __init__(self, *, tap_layer: str) -> None:
        self.tap_layer = tap_layer
        self._gates: dict[str, RouterGate] = {}

    @property
    def registered_units(self) -> tuple[str, ...]:
        return tuple(self._gates.keys())

    def register(self, gate: RouterGate) -> None:
        if gate.tap_layer != self.tap_layer:
            raise ValueError(
                f"Gate '{gate.unit_id}' uses tap_layer '{gate.tap_layer}', "
                f"expected '{self.tap_layer}'"
            )
        self._gates[gate.unit_id] = gate

    def unregister(self, unit_id: str) -> RouterGate | None:
        return self._gates.pop(unit_id, None)

    def get_gate(self, unit_id: str) -> RouterGate:
        return self._gates[unit_id]

    def route(self, embedding: np.ndarray, *, top_k: int = 1) -> RoutingDecision:
        """Select up to top_k matching Skill Units ranked by margin above threshold."""
        if top_k <= 0:
            raise ValueError("top_k must be >= 1")
        scores: dict[str, float] = {}
        thresholds: dict[str, float] = {}
        passing: list[tuple[float, str]] = []

        for unit_id, gate in self._gates.items():
            sc = gate.score(embedding)
            scores[unit_id] = sc
            thresholds[unit_id] = gate.threshold
            if sc >= gate.threshold:
                passing.append((sc - gate.threshold, unit_id))

        passing.sort(key=lambda pair: pair[0], reverse=True)
        selected = tuple(uid for _, uid in passing[:top_k])
        return RoutingDecision(
            selected_units=selected,
            scores=scores,
            thresholds=thresholds,
        )
