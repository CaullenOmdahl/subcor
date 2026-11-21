"""Tests for text-level Intern telemetry store and n-gram repetition metrics."""

from __future__ import annotations

from pathlib import Path

import pytest

from slaingi.intern import TextTrajectory, TextTrajectoryStore, ngram_repetition_ratio
from slaingi.types import VerificationError, VerificationEvidence


def test_ngram_repetition_ratio_distinguishes_looping_from_unique_text() -> None:
    clean = "The quick brown fox jumps over the lazy dog near the quiet river bank."
    looping = " ".join(["wait let me recheck step one"] * 12)
    assert ngram_repetition_ratio(clean) == 0.0
    assert ngram_repetition_ratio(looping) > 0.75


def test_text_trajectory_store_enforces_verified_evidence_gate(tmp_path: Path) -> None:
    store = TextTrajectoryStore(tmp_path / "intern.jsonl")
    store.append_many(
        [
            TextTrajectory(
                trajectory_id="ok-1",
                task_family="ops",
                seat="intern",
                prompt="Event: invoice over cap",
                response="wrong",
                tokens_generated=4,
                hit_token_limit=False,
                repetition_ratio=0.0,
                passed_verifier=False,
                evidence=VerificationEvidence(
                    verified=True,
                    verifier="senior_ceo",
                    method="senior_seat_review",
                ),
                senior_correction='{"seat":"finance","gate":"owner","code":"FIN-CAP"}',
                layer_activations={"layers.15.mlp.down_proj": [1.0, 2.0, 3.0]},
            ),
            TextTrajectory(
                trajectory_id="unverified-1",
                task_family="ops",
                seat="intern",
                prompt="Event: poisoned",
                response="bad",
                tokens_generated=2,
                hit_token_limit=False,
                repetition_ratio=0.0,
                passed_verifier=False,
                evidence=VerificationEvidence(
                    verified=False,
                    verifier="none",
                    method="unverified_draft",
                ),
                senior_correction='{"seat":"bad"}',
                layer_activations={"layers.15.mlp.down_proj": [9.0, 9.0, 9.0]},
            ),
        ]
    )

    assert len(store.load_all()) == 2
    assert len(store.load_verified()) == 1
    pairs = store.verified_distillation_pairs(task_family="ops")
    assert pairs == [
        ("Event: invoice over cap", '{"seat":"finance","gate":"owner","code":"FIN-CAP"}')
    ]

    with pytest.raises(VerificationError):
        store.verified_distillation_pairs(task_family="unknown")
