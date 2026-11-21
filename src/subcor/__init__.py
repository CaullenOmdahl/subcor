"""SUBCOR: Sub-Bit Unit-Based Continual On-Policy Runtime.

Governed weight-space surgery, function-preserving dynamic low-rank expansion,
and sub-1-bit binary Skill Units over frozen quantized base kernels.
"""

from subcor.benchmark import (
    AxisSummary,
    BenchmarkCase,
    BenchmarkLedger,
    BenchmarkSnapshot,
    BenchmarkSuite,
    CaseResult,
    GrowthDelta,
    compare_snapshots,
    core_v1_suite,
    grade_response,
    render_markdown_report,
    run_benchmark_http,
    run_benchmark_local,
)
from subcor.confidence import (
    ConfidencePolicy,
    TurnConfidence,
    evaluate_turn_confidence,
    parse_tool_calls,
)
from subcor.expansion import ElasticLinear, LowRankAdapter
from subcor.proving_ground import MicroInternNet, ProvingGroundReport, run_proving_ground
from subcor.recompile import RecompiledUnitOutcome, RecompileReport, recompile_units_for_kernel
from subcor.router import MoESkillRouter, RouterGate, RoutingDecision
from subcor.sub1bit import Sub1BitMatrix, factorize_sub1bit, relative_frobenius_error
from subcor.surgeon import LayerDiagnosis, SurgicalPatch, WeightSpaceSurgeon
from subcor.telemetry import TrajectoryStore
from subcor.types import (
    AdaptationMode,
    ConformanceReport,
    DeltaEncoding,
    GovernanceError,
    IntegrityError,
    SkillSpec,
    SkillUnitManifest,
    SubcorError,
    TrajectoryRecord,
    UnitState,
    VerificationError,
    VerificationEvidence,
)
from subcor.units import LayerDeltaPayload, SkillUnitRegistry, build_layer_delta

__version__ = "0.1.0"

__all__ = [
    "AdaptationMode",
    "AxisSummary",
    "BenchmarkCase",
    "BenchmarkLedger",
    "BenchmarkSnapshot",
    "BenchmarkSuite",
    "CaseResult",
    "ConfidencePolicy",
    "ConformanceReport",
    "DeltaEncoding",
    "ElasticLinear",
    "GovernanceError",
    "GrowthDelta",
    "IntegrityError",
    "LayerDeltaPayload",
    "LayerDiagnosis",
    "LowRankAdapter",
    "MicroInternNet",
    "MoESkillRouter",
    "ProvingGroundReport",
    "RecompileReport",
    "RecompiledUnitOutcome",
    "RouterGate",
    "RoutingDecision",
    "SkillSpec",
    "SkillUnitManifest",
    "SkillUnitRegistry",
    "SubcorError",
    "Sub1BitMatrix",
    "SurgicalPatch",
    "TrajectoryRecord",
    "TrajectoryStore",
    "TurnConfidence",
    "UnitState",
    "VerificationError",
    "VerificationEvidence",
    "WeightSpaceSurgeon",
    "__version__",
    "build_layer_delta",
    "compare_snapshots",
    "core_v1_suite",
    "evaluate_turn_confidence",
    "factorize_sub1bit",
    "grade_response",
    "parse_tool_calls",
    "recompile_units_for_kernel",
    "relative_frobenius_error",
    "render_markdown_report",
    "run_benchmark_http",
    "run_benchmark_local",
    "run_proving_ground",
]
