"""Governed Skill Unit registry, integrity checks, rollback, and CaPheOS export.

Every learned adapter or surgical weight patch is packaged as an immutable,
content-addressed Skill Unit with an explicit lifecycle:

    PROPOSED -> DORMANT -> CONFORMANT -> ENABLED -> ROLLED_BACK

Governance invariants enforced in code:
  1. Payload SHA-256 digest verification on load/enable (fails closed on tampering).
  2. Locked Conformance Gate: must preserve baseline score within tolerance and
     strictly improve the target defect/skill metric.
  3. No self-approval: approver identity must differ from proposer identity.
  4. Atomic rollback: disabling or rolling back an enabled unit cleanly removes
     its delta stack without touching the frozen base weights.
  5. Portable CaPheOS export: emits a valid format-2 module.json bundle with
     declared capabilities, conformance spec, and zero company data leakage.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from slaingi.expansion import LowRankAdapter
from slaingi.router import RouterGate
from slaingi.sub1bit import Sub1BitMatrix
from slaingi.types import (
    AdaptationMode,
    ConformanceReport,
    DeltaEncoding,
    GovernanceError,
    IntegrityError,
    SkillUnitManifest,
    UnitState,
)

_UNIT_MAGIC = b"SLUN"
_UNIT_VERSION = 1
_ENC_LOW_RANK_FP32 = 1
_ENC_SUB1BIT = 2


@dataclass(frozen=True)
class LayerDeltaPayload:
    """Serialized delta for a single named layer inside a Skill Unit."""

    layer_name: str
    encoding: DeltaEncoding
    sub1bit: Sub1BitMatrix | None = None
    a_fp32: np.ndarray | None = None  # (rank, d_in) float32
    b_fp32: np.ndarray | None = None  # (d_out, rank) float32
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.encoding is DeltaEncoding.SUB1BIT_BINARY and self.sub1bit is None:
            raise ValueError("SUB1BIT_BINARY payload requires sub1bit matrix")
        if self.encoding is DeltaEncoding.LOW_RANK_FP32 and (
            self.a_fp32 is None or self.b_fp32 is None
        ):
            raise ValueError("LOW_RANK_FP32 payload requires a_fp32 and b_fp32")

    @property
    def num_weights(self) -> int:
        if self.encoding is DeltaEncoding.SUB1BIT_BINARY:
            assert self.sub1bit is not None
            return self.sub1bit.num_weights
        assert self.a_fp32 is not None and self.b_fp32 is not None
        return self.b_fp32.shape[0] * self.a_fp32.shape[1]

    @property
    def payload_bits(self) -> int:
        if self.encoding is DeltaEncoding.SUB1BIT_BINARY:
            assert self.sub1bit is not None
            return self.sub1bit.payload_bits
        assert self.a_fp32 is not None and self.b_fp32 is not None
        return 32 * (self.a_fp32.size + self.b_fp32.size) + 32

    def to_adapter(self, adapter_name: str) -> LowRankAdapter:
        """Materialize a frozen hot-swappable LowRankAdapter for runtime mounting."""
        if self.encoding is DeltaEncoding.SUB1BIT_BINARY:
            assert self.sub1bit is not None
            return LowRankAdapter.from_sub1bit(adapter_name, self.sub1bit)
        assert self.a_fp32 is not None and self.b_fp32 is not None
        return LowRankAdapter(
            name=adapter_name,
            a=self.a_fp32.astype(np.float64),
            b=self.b_fp32.astype(np.float64),
            scale=self.scale,
            binary_ste=False,
            frozen=True,
        )

    def encode_blob(self) -> bytes:
        if self.encoding is DeltaEncoding.SUB1BIT_BINARY:
            assert self.sub1bit is not None
            return self.sub1bit.to_bytes()
        assert self.a_fp32 is not None and self.b_fp32 is not None
        d_out, rank = self.b_fp32.shape
        rank_a, d_in = self.a_fp32.shape
        if rank != rank_a:
            raise ValueError("Rank mismatch in LOW_RANK_FP32 payload")
        header = struct.pack("<IIIf", d_out, d_in, rank, float(self.scale))
        return header + self.a_fp32.astype("<f4").tobytes() + self.b_fp32.astype("<f4").tobytes()

    @classmethod
    def decode_blob(cls, layer_name: str, encoding_code: int, blob: bytes) -> LayerDeltaPayload:
        if encoding_code == _ENC_SUB1BIT:
            return cls(
                layer_name=layer_name,
                encoding=DeltaEncoding.SUB1BIT_BINARY,
                sub1bit=Sub1BitMatrix.from_bytes(blob),
            )
        if encoding_code == _ENC_LOW_RANK_FP32:
            hdr_size = struct.calcsize("<IIIf")
            if len(blob) < hdr_size:
                raise IntegrityError("Truncated LOW_RANK_FP32 layer blob")
            d_out, d_in, rank, scale = struct.unpack_from("<IIIf", blob, 0)
            a_bytes = rank * d_in * 4
            b_bytes = d_out * rank * 4
            if len(blob) != hdr_size + a_bytes + b_bytes:
                raise IntegrityError("Corrupt LOW_RANK_FP32 layer blob length")
            a = (
                np.frombuffer(blob[hdr_size : hdr_size + a_bytes], dtype="<f4")
                .reshape((rank, d_in))
                .copy()
            )
            b = np.frombuffer(blob[hdr_size + a_bytes :], dtype="<f4").reshape((d_out, rank)).copy()
            return cls(
                layer_name=layer_name,
                encoding=DeltaEncoding.LOW_RANK_FP32,
                a_fp32=a,
                b_fp32=b,
                scale=float(scale),
            )
        raise IntegrityError(f"Unknown layer delta encoding code: {encoding_code}")


def serialize_layer_deltas(deltas: list[LayerDeltaPayload]) -> bytes:
    """Encode a list of LayerDeltaPayloads into a deterministic binary bundle."""
    parts = [struct.pack("<4sBH", _UNIT_MAGIC, _UNIT_VERSION, len(deltas))]
    for delta in deltas:
        name_bytes = delta.layer_name.encode("utf-8")
        enc_code = (
            _ENC_SUB1BIT if delta.encoding is DeltaEncoding.SUB1BIT_BINARY else _ENC_LOW_RANK_FP32
        )
        blob = delta.encode_blob()
        parts.append(struct.pack("<H", len(name_bytes)))
        parts.append(name_bytes)
        parts.append(struct.pack("<BI", enc_code, len(blob)))
        parts.append(blob)
    return b"".join(parts)


def deserialize_layer_deltas(raw: bytes) -> list[LayerDeltaPayload]:
    """Decode a binary bundle produced by serialize_layer_deltas()."""
    hdr = struct.Struct("<4sBH")
    if len(raw) < hdr.size:
        raise IntegrityError("Truncated Skill Unit payload")
    magic, version, count = hdr.unpack_from(raw, 0)
    if magic != _UNIT_MAGIC:
        raise IntegrityError(f"Invalid Skill Unit payload magic: {magic!r}")
    if version != _UNIT_VERSION:
        raise IntegrityError(f"Unsupported Skill Unit payload version: {version}")

    offset = hdr.size
    deltas: list[LayerDeltaPayload] = []
    for _ in range(count):
        if offset + 2 > len(raw):
            raise IntegrityError("Truncated layer name header")
        (name_len,) = struct.unpack_from("<H", raw, offset)
        offset += 2
        if offset + name_len + 5 > len(raw):
            raise IntegrityError("Truncated layer delta entry")
        layer_name = raw[offset : offset + name_len].decode("utf-8")
        offset += name_len
        enc_code, blob_len = struct.unpack_from("<BI", raw, offset)
        offset += 5
        if offset + blob_len > len(raw):
            raise IntegrityError("Truncated layer delta blob")
        blob = raw[offset : offset + blob_len]
        offset += blob_len
        deltas.append(LayerDeltaPayload.decode_blob(layer_name, enc_code, blob))

    if offset != len(raw):
        raise IntegrityError(f"Unexpected trailing payload bytes: {len(raw) - offset}")
    return deltas


def build_layer_delta(
    layer_name: str,
    adapter: LowRankAdapter,
    *,
    encoding: DeltaEncoding = DeltaEncoding.SUB1BIT_BINARY,
    sub1bit_rank: int | None = None,
) -> LayerDeltaPayload:
    """Convert an in-memory LowRankAdapter into a serializable LayerDeltaPayload."""
    if encoding is DeltaEncoding.SUB1BIT_BINARY:
        target_rank = adapter.rank if sub1bit_rank is None else sub1bit_rank
        mat = adapter.compress_sub1bit(rank=target_rank)
        return LayerDeltaPayload(
            layer_name=layer_name,
            encoding=DeltaEncoding.SUB1BIT_BINARY,
            sub1bit=mat,
        )
    if adapter.binary_ste:
        # Materialize and re-factor via exact SVD when exporting binary_ste as FP32 low-rank.
        u, s, vt = np.linalg.svd(adapter.delta_matrix(), full_matrices=False)
        r = min(adapter.rank, len(s))
        b = (u[:, :r] * s[:r]).astype(np.float32)
        a = vt[:r, :].astype(np.float32)
        return LayerDeltaPayload(
            layer_name=layer_name,
            encoding=DeltaEncoding.LOW_RANK_FP32,
            a_fp32=a,
            b_fp32=b,
            scale=1.0,
        )
    return LayerDeltaPayload(
        layer_name=layer_name,
        encoding=DeltaEncoding.LOW_RANK_FP32,
        a_fp32=adapter.a.astype(np.float32),
        b_fp32=adapter.b.astype(np.float32),
        scale=float(adapter.scale),
    )


class SkillUnitRegistry:
    """Filesystem-backed, content-addressed governed registry for Skill Units."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _unit_dir(self, unit_id: str) -> Path:
        if "/" in unit_id or "\\" in unit_id or ".." in unit_id or not unit_id.strip():
            raise ValueError(f"Invalid unit_id: {unit_id!r}")
        return self.root / unit_id

    def _manifest_path(self, unit_id: str) -> Path:
        return self._unit_dir(unit_id) / "manifest.json"

    def _payload_path(self, unit_id: str) -> Path:
        return self._unit_dir(unit_id) / "payload.slun"

    def _router_gate_path(self, unit_id: str) -> Path:
        return self._unit_dir(unit_id) / "router_gate.json"

    def save_router_gate(self, gate: RouterGate) -> None:
        """Attach or update a persistent MoE RouterGate for an existing Skill Unit."""
        self.get_manifest(gate.unit_id)
        self._router_gate_path(gate.unit_id).write_text(
            json.dumps(gate.to_dict(), indent=2) + "\n",
            encoding="utf-8",
        )

    def load_router_gate(self, unit_id: str) -> RouterGate | None:
        """Load the persistent MoE RouterGate for a Skill Unit, if present."""
        path = self._router_gate_path(unit_id)
        if not path.exists():
            return None
        return RouterGate.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _write_manifest(self, manifest: SkillUnitManifest) -> None:
        path = self._manifest_path(manifest.unit_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(manifest.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def propose(
        self,
        *,
        unit_id: str,
        version: str,
        title: str,
        summary: str,
        proposer: str,
        mode: AdaptationMode,
        deltas: list[LayerDeltaPayload],
        license_id: str = "Apache-2.0",
    ) -> SkillUnitManifest:
        """Package proposed layer deltas into a content-addressed DORMANT Skill Unit."""
        if not deltas:
            raise ValueError("A Skill Unit must contain at least one LayerDeltaPayload")
        unit_dir = self._unit_dir(unit_id)
        if unit_dir.exists():
            raise GovernanceError(f"Skill Unit '{unit_id}' already exists; bump unit_id/version")

        raw_payload = serialize_layer_deltas(deltas)
        digest = hashlib.sha256(raw_payload).hexdigest()
        total_bits = sum(d.payload_bits for d in deltas)
        total_weights = sum(d.num_weights for d in deltas)
        bpw = total_bits / total_weights if total_weights > 0 else 0.0

        encodings = {d.encoding for d in deltas}
        unit_encoding = (
            next(iter(encodings)) if len(encodings) == 1 else DeltaEncoding.LOW_RANK_FP32
        )

        manifest = SkillUnitManifest(
            unit_id=unit_id,
            version=version,
            title=title,
            summary=summary,
            proposer=proposer,
            mode=mode,
            encoding=unit_encoding,
            target_layers=tuple(d.layer_name for d in deltas),
            payload_sha256=digest,
            payload_bytes=len(raw_payload),
            effective_bits_per_weight=bpw,
            state=UnitState.DORMANT,
            license=license_id,
        )
        unit_dir.mkdir(parents=True, exist_ok=False)
        self._payload_path(unit_id).write_bytes(raw_payload)
        self._write_manifest(manifest)
        return manifest

    def get_manifest(self, unit_id: str) -> SkillUnitManifest:
        path = self._manifest_path(unit_id)
        if not path.exists():
            raise FileNotFoundError(f"Unknown Skill Unit '{unit_id}'")
        return SkillUnitManifest.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def list_units(self) -> list[SkillUnitManifest]:
        manifests: list[SkillUnitManifest] = []
        if not self.root.exists():
            return manifests
        for child in sorted(self.root.iterdir()):
            mpath = child / "manifest.json"
            if child.is_dir() and mpath.exists():
                manifests.append(SkillUnitManifest.from_dict(json.loads(mpath.read_text("utf-8"))))
        return manifests

    def load_verified_deltas(self, unit_id: str) -> list[LayerDeltaPayload]:
        """Verify SHA-256 integrity of payload on disk and decode its layer deltas."""
        manifest = self.get_manifest(unit_id)
        payload_path = self._payload_path(unit_id)
        if not payload_path.exists():
            raise IntegrityError(f"Missing payload for Skill Unit '{unit_id}'")
        raw = payload_path.read_bytes()
        actual_sha = hashlib.sha256(raw).hexdigest()
        if actual_sha != manifest.payload_sha256:
            raise IntegrityError(
                f"SHA-256 mismatch for Skill Unit '{unit_id}': "
                f"expected {manifest.payload_sha256}, got {actual_sha}"
            )
        return deserialize_layer_deltas(raw)

    def record_conformance(
        self,
        unit_id: str,
        *,
        baseline_before: float,
        baseline_after: float,
        target_before: float,
        target_after: float,
        max_regression_tolerance: float = 0.02,
        min_target_improvement: float = 0.05,
        evaluator: str = "locked_conformance_gate",
    ) -> SkillUnitManifest:
        """Evaluate locked conformance rules and transition DORMANT -> CONFORMANT on pass."""
        manifest = self.get_manifest(unit_id)
        # Verify integrity before certifying conformance.
        self.load_verified_deltas(unit_id)

        regression = baseline_before - baseline_after
        improvement = target_after - target_before

        if regression > max_regression_tolerance:
            passed = False
            reason = (
                f"Baseline regression {regression:.4f} exceeded tolerance "
                f"{max_regression_tolerance:.4f} "
                f"({baseline_before:.4f} -> {baseline_after:.4f})"
            )
        elif improvement < min_target_improvement:
            passed = False
            reason = (
                f"Target improvement {improvement:.4f} below minimum "
                f"{min_target_improvement:.4f} "
                f"({target_before:.4f} -> {target_after:.4f})"
            )
        else:
            passed = True
            reason = (
                f"Passed: baseline {baseline_before:.4f} -> {baseline_after:.4f}, "
                f"target {target_before:.4f} -> {target_after:.4f} (+{improvement:.4f})"
            )

        report = ConformanceReport(
            unit_id=unit_id,
            passed=passed,
            baseline_before=baseline_before,
            baseline_after=baseline_after,
            target_before=target_before,
            target_after=target_after,
            max_regression_tolerance=max_regression_tolerance,
            min_target_improvement=min_target_improvement,
            reason=reason,
            evaluator=evaluator,
        )
        new_state = UnitState.CONFORMANT if passed else UnitState.DORMANT
        updated = replace(manifest, state=new_state, conformance=report)
        self._write_manifest(updated)
        return updated

    def promote(self, unit_id: str, *, approver: str) -> SkillUnitManifest:
        """Promote a CONFORMANT unit to ENABLED.

        Enforces:
          - Payload SHA-256 verification.
          - Conformance gate must have passed.
          - Separation of duties: approver must differ from proposer (no self-approval).
        """
        manifest = self.get_manifest(unit_id)
        if not approver or not approver.strip():
            raise GovernanceError("Approver identity is required for promotion")
        if approver.strip() == manifest.proposer.strip():
            raise GovernanceError(
                f"Self-approval forbidden: proposer '{manifest.proposer}' "
                f"cannot promote unit '{unit_id}'"
            )
        if manifest.state is not UnitState.CONFORMANT or (
            manifest.conformance is None or not manifest.conformance.passed
        ):
            raise GovernanceError(
                f"Unit '{unit_id}' is in state '{manifest.state.value}' and has not passed "
                f"conformance; promotion denied"
            )
        # Re-verify digest right before promotion.
        self.load_verified_deltas(unit_id)
        updated = replace(manifest, state=UnitState.ENABLED, approver=approver.strip())
        self._write_manifest(updated)
        return updated

    def rollback(self, unit_id: str, *, actor: str, reason: str = "") -> SkillUnitManifest:
        """Atomically mark an ENABLED unit as ROLLED_BACK."""
        manifest = self.get_manifest(unit_id)
        if manifest.state is not UnitState.ENABLED:
            raise GovernanceError(
                f"Cannot roll back unit '{unit_id}' from state '{manifest.state.value}'"
            )
        updated = replace(manifest, state=UnitState.ROLLED_BACK)
        self._write_manifest(updated)

        audit_log = self._unit_dir(unit_id) / "audit.jsonl"
        entry = {
            "event": "rollback",
            "unit_id": unit_id,
            "actor": actor,
            "reason": reason,
        }
        with audit_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, sort_keys=True) + "\n")
        return updated

    def export_capheos_unit(self, unit_id: str, destination_root: Path) -> Path:
        """Export a CONFORMANT or ENABLED Skill Unit as a CaPheOS format-2 unit bundle.

        Follows CaPheOS §5 Unit Standard:
          - Portable capability manifest (`module.json`, format 2)
          - Content-addressed payload + SHA-256 verification metadata
          - Scoped read-only filesystem capabilities; no company memory or secrets
          - Conformance check script (`conformance/test.js`) verifying digest & report
        """
        manifest = self.get_manifest(unit_id)
        if manifest.state not in (UnitState.CONFORMANT, UnitState.ENABLED):
            raise GovernanceError(
                f"Only CONFORMANT or ENABLED units can be exported (got '{manifest.state.value}')"
            )
        # Verify integrity before export.
        self.load_verified_deltas(unit_id)

        out_dir = destination_root / unit_id
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "conformance").mkdir(exist_ok=True)

        payload_src = self._payload_path(unit_id)
        payload_dst = out_dir / "weights.slun"
        payload_dst.write_bytes(payload_src.read_bytes())

        gate = self.load_router_gate(unit_id)
        router_gate_file: str | None = None
        if gate is not None:
            router_gate_file = "router_gate.json"
            (out_dir / router_gate_file).write_text(
                json.dumps(gate.to_dict(), indent=2) + "\n",
                encoding="utf-8",
            )

        capheos_manifest: dict[str, Any] = {
            "format": 2,
            "id": f"slaingi-{manifest.unit_id}",
            "version": manifest.version,
            "kind": "capability",
            "title": manifest.title,
            "summary": manifest.summary,
            "license": manifest.license,
            "engines": {"caPheOS": ">=0.2", "node": ">=22"},
            "requires": {
                "coreModules": ["observe", "verify"],
                "guardrails": ["reviewWithoutSelfApproval"],
            },
            "capabilities": {
                "network": [],
                "fs": {
                    "read": [f"company/work/units/slaingi-{manifest.unit_id}/**"],
                    "write": [],
                },
                "secrets": [],
                "subprocess": False,
            },
            "slaingi": {
                "mode": manifest.mode.value,
                "encoding": manifest.encoding.value,
                "targetLayers": list(manifest.target_layers),
                "payloadFile": "weights.slun",
                "routerGateFile": router_gate_file,
                "payloadSha256": manifest.payload_sha256,
                "payloadBytes": manifest.payload_bytes,
                "effectiveBitsPerWeight": round(manifest.effective_bits_per_weight, 4),
                "proposer": manifest.proposer,
                "approver": manifest.approver,
                "conformance": (
                    None if manifest.conformance is None else manifest.conformance.to_dict()
                ),
            },
            "conformance": {"test": "conformance/test.js"},
        }
        (out_dir / "module.json").write_text(
            json.dumps(capheos_manifest, indent=2) + "\n",
            encoding="utf-8",
        )

        # Standalone Node >=22 conformance test verifying payload digest & passed gate.
        test_js = """import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const root = dirname(dirname(fileURLToPath(import.meta.url)));
const manifest = JSON.parse(readFileSync(join(root, 'module.json'), 'utf8'));
const payload = readFileSync(join(root, manifest.slaingi.payloadFile));
const actualSha = createHash('sha256').update(payload).digest('hex');

if (manifest.format !== 2) {
  throw new Error(`Expected CaPheOS unit format 2, got ${manifest.format}`);
}
if (actualSha !== manifest.slaingi.payloadSha256) {
  throw new Error(`Payload SHA-256 mismatch: expected ${manifest.slaingi.payloadSha256}, got ${actualSha}`);
}
if (!manifest.slaingi.conformance || manifest.slaingi.conformance.passed !== true) {
  throw new Error('Exported SLAINGI unit must carry a passing conformance report');
}
console.log(`PASS ${manifest.id}@${manifest.version} sha256=${actualSha.slice(0, 12)} bpw=${manifest.slaingi.effectiveBitsPerWeight}`);
"""
        (out_dir / "conformance" / "test.js").write_text(test_js, encoding="utf-8")
        return out_dir
