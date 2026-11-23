"""SMOKE-TEST FIXTURE ONLY: a SUBCOR tenant with one ENABLED unit whose RouterGate is fit
on real frozen-base embeddings, so the router probe exercises a live gate end to end
without a (GPU) Study Cycle.

The unit's payload is `build_dummy_delta` (no real LoRA). Against an external
llama-server SUBCOR then warns "no adapter" and decodes on the base, which is fine for a
routing probe. Do NOT use this tenant for capability numbers; the real probe runs against
the tenant holding the trained tool-use units.

Gate fit: ~20 hand-written tool-call style positives vs general negatives (the data
agent's neg_prompts.txt if given, plus a built-in handful), all disjoint from the probe
set. Embeddings come from LlamaCppServingAdapter.encode_prompt_embedding, i.e. exactly
the path SUBCOR serve routes with (llama-server --embeddings --pooling mean).

Run in the PROJECT env (needs subcor):
  cd <repo> && uv run --extra dev --extra rocm python experiments/bfcl/general/make_probe_fixture.py \
      --llamacpp-url http://127.0.0.1:8502 --state-dir <scratch>/general/subcor_state --tenant probe
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from subcor.backends.llamacpp_kernel import LlamaCppServingAdapter
from subcor.router import RouterGate
from subcor.server import build_dummy_delta
from subcor.types import AdaptationMode
from subcor.units import SkillUnitRegistry

POSITIVES = [
    "Use the get_weather tool to find the current temperature in Berlin in celsius.",
    "Call the calculate_bmi function with weight 70 kg and height 175 cm.",
    "Book a table for 4 at Luigi's tonight at 7pm using the restaurant reservation API.",
    "Find flights from SFO to JFK on December 3 using the flight_search tool.",
    "Send an email to alice@example.com with subject 'Meeting' via the send_email function.",
    "Get the stock price of AAPL using the finance API.",
    "Convert 250 USD to EUR with the currency_converter tool.",
    "Use the calendar tool to create an event called standup tomorrow at 9am.",
    "Call get_movie_details for the film Inception.",
    "Look up the population of Canada using the country_info function.",
    "Run the translate tool to translate 'good night' into Japanese.",
    "Fetch the latest 5 news headlines about climate using news_api.get_headlines.",
    "Use the math.factorial function to compute the factorial of 12.",
    "Query the database tool for all users created after 2024-01-01.",
    "Call the geocode function for the address 1600 Amphitheatre Parkway.",
    "Use the timer tool to set a 25 minute timer.",
    "Invoke the spotify.play function to play 'Bohemian Rhapsody'.",
    "Call the uber.ride function to request a ride to the airport for 2 people.",
    "Use the get_distance tool to compute the distance between Paris and Lyon in km.",
    "With the shopping API, add two packs of AA batteries to my cart.",
]
BUILTIN_NEGATIVES = [
    "Tell me a joke about cats.",
    "What causes the seasons on Earth?",
    "Write a short poem about the ocean.",
    "Who wrote Pride and Prejudice?",
    "Give me three ideas for a birthday party.",
    "Explain photosynthesis to a ten-year-old.",
    "What is the boiling point of water at sea level?",
    "Describe your ideal weekend in one paragraph.",
]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--llamacpp-url", default="http://127.0.0.1:8502")
    ap.add_argument("--model-id", default="fixture")
    ap.add_argument("--state-dir", type=Path, required=True)
    ap.add_argument("--tenant", default="probe")
    ap.add_argument("--unit-id", default="fixture-tooluse-gate")
    ap.add_argument("--neg-prompts", type=Path, default=None)
    args = ap.parse_args()

    negs = list(BUILTIN_NEGATIVES)
    if args.neg_prompts and args.neg_prompts.exists():
        negs += [ln.strip() for ln in args.neg_prompts.read_text().splitlines() if ln.strip()]

    kernel = LlamaCppServingAdapter(base_url=args.llamacpp_url, model_id=args.model_id)
    pos = np.stack([kernel.encode_prompt_embedding(p) for p in POSITIVES])
    neg = np.stack([kernel.encode_prompt_embedding(p) for p in negs])
    gate = RouterGate.fit(
        unit_id=args.unit_id, tap_layer=kernel.tap_layer, pos_embeddings=pos, neg_embeddings=neg
    )

    reg = SkillUnitRegistry(
        args.state_dir.expanduser().resolve() / "tenants" / args.tenant / "units"
    )
    if (reg.root / args.unit_id).exists():
        raise SystemExit(
            f"unit {args.unit_id} already exists under {reg.root}; use a fresh --state-dir"
        )
    reg.propose(
        unit_id=args.unit_id,
        version="0.0.1",
        title="probe fixture (dummy payload)",
        summary="Smoke-test fixture: real embedding gate, dummy delta. Not a capability unit.",
        proposer="general-eval-fixture",
        mode=AdaptationMode.MANUAL_TEST,
        deltas=[build_dummy_delta("blk.0.ffn_down")],
        base_kernel_id=args.model_id,
    )
    reg.save_router_gate(gate)
    reg.record_conformance(
        args.unit_id,
        baseline_before=1.0,
        baseline_after=1.0,
        target_before=0.0,
        target_after=1.0,
        evaluator="fixture-not-a-real-gate",
    )
    reg.promote(args.unit_id, approver="general-eval-fixture-approver")
    print(
        f"fixture unit '{args.unit_id}' ENABLED for tenant '{args.tenant}' "
        f"(dim={gate.dim}, threshold={gate.threshold:.4f}, pos_mean={gate.pos_mean_score:.4f}, "
        f"neg_mean={gate.neg_mean_score:.4f}, n_pos={len(POSITIVES)}, n_neg={len(negs)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
