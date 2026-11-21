"""Role- and tenant-scoped authentication and authorization for SUBCOR HTTP server.

Addresses Issue #13:
  - Binds caller credentials to explicit actor identity, allowed roles, and allowed tenants.
  - Prevents self-asserted separation of duties (`approver != proposer`) by binding
    `proposer` / `approver` to the authenticated principal's actor identity.
  - Prevents unverified clients from spoofing `evidence.verified = true`.
  - Preserves simple single-key admin mode (`AuthPolicy.from_single_key(...)`) for
    single-operator loopback / SSH-tunnel deployments.
"""

from __future__ import annotations

import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ALL_ROLES: frozenset[str] = frozenset(
    {"chat", "ingest", "verify", "propose", "approve", "rollback", "export", "admin"}
)


@dataclass(frozen=True)
class AuthPrincipal:
    """Authenticated caller identity with scoped roles and tenants."""

    actor: str
    roles: frozenset[str]
    tenants: frozenset[str] = frozenset({"*"})

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles

    def has_role(self, role: str) -> bool:
        return self.is_admin or (role in self.roles)

    def allows_tenant(self, tenant_id: str) -> bool:
        return "*" in self.tenants or tenant_id in self.tenants

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AuthPrincipal:
        roles = frozenset(str(r) for r in data.get("roles", []))
        tenants = frozenset(str(t) for t in data.get("tenants", ["*"]))
        unknown = roles - ALL_ROLES
        if unknown:
            raise ValueError(f"Unknown auth roles: {sorted(unknown)}")
        return cls(
            actor=str(data["actor"]),
            roles=roles,
            tenants=tenants,
        )


class AuthPolicy:
    """Token-to-AuthPrincipal registry supporting single-key and multi-role RBAC modes."""

    def __init__(self, tokens: dict[str, AuthPrincipal] | None = None) -> None:
        self._tokens: dict[str, AuthPrincipal] = dict(tokens or {})

    @property
    def enabled(self) -> bool:
        return bool(self._tokens)

    @classmethod
    def disabled(cls) -> AuthPolicy:
        return cls({})

    @classmethod
    def from_single_key(cls, api_key: str, *, actor: str = "admin") -> AuthPolicy:
        return cls({api_key: AuthPrincipal(actor=actor, roles=frozenset({"admin"}))})

    @classmethod
    def from_json_file(cls, path: Path) -> AuthPolicy:
        raw = json.loads(path.read_text(encoding="utf-8"))
        entries = raw.get("tokens", raw) if isinstance(raw, dict) else []
        tokens: dict[str, AuthPrincipal] = {}
        if isinstance(entries, dict):
            for tok, spec in entries.items():
                tokens[str(tok)] = AuthPrincipal.from_dict(spec)
        elif isinstance(entries, list):
            for item in entries:
                tokens[str(item["token"])] = AuthPrincipal.from_dict(item)
        else:
            raise ValueError("auth config must contain a 'tokens' mapping or list")
        return cls(tokens)

    def authenticate(self, token: str | None) -> AuthPrincipal | None:
        """Constant-time token lookup returning the matched AuthPrincipal or None."""
        if not self.enabled:
            return AuthPrincipal(actor="anonymous-dev", roles=frozenset({"admin"}))
        if not token:
            return None
        matched: AuthPrincipal | None = None
        for stored_tok, principal in self._tokens.items():
            if hmac.compare_digest(token, stored_tok):
                matched = principal
        return matched
