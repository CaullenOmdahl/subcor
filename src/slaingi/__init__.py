"""SLAINGI: Self-Learning AI General Intelligence.

Governed weight-space surgery, function-preserving dynamic low-rank expansion,
and sub-1-bit binary Skill Units over frozen quantized base kernels.
"""

from slaingi.expansion import ElasticLinear, LowRankAdapter
from slaingi.proving_ground import MicroInternNet, ProvingGroundReport, run_proving_ground
from slaingi.router import MoESkillRouter, RouterGate, RoutingDecision
from slaingi.sub1bit import Sub1BitMatrix, factorize_sub1bit, relative_frobenius_error
from slaingi.surgeon import LayerDiagnosis, SurgicalPatch, WeightSpaceSurgeon
from slaingi.telemetry import TrajectoryStore
from slaingi.types import (
    AdaptationMode,
    ConformanceReport,
    DeltaEncoding,
    GovernanceError,
    IntegrityError,
    SkillUnitManifest,
    SlaingiError,
    TrajectoryRecord,
    UnitState,
    VerificationError,
    VerificationEvidence,
)
from slaingi.units import LayerDeltaPayload, SkillUnitRegistry, build_layer_delta

__version__ = "0.1.0"

__all__ = [
    "AdaptationMode",
    "ConformanceReport",
    "DeltaEncoding",
    "ElasticLinear",
    "GovernanceError",
    "IntegrityError",
    "LayerDeltaPayload",
    "LayerDiagnosis",
    "LowRankAdapter",
    "MicroInternNet",
    "MoESkillRouter",
    "ProvingGroundReport",
    "RouterGate",
    "RoutingDecision",
    "SkillUnitManifest",
    "SkillUnitRegistry",
    "SlaingiError",
    "Sub1BitMatrix",
    "SurgicalPatch",
    "TrajectoryRecord",
    "TrajectoryStore",
    "UnitState",
    "VerificationError",
    "VerificationEvidence",
    "WeightSpaceSurgeon",
    "__version__",
    "build_layer_delta",
    "factorize_sub1bit",
    "relative_frobenius_error",
    "run_proving_ground",
]
