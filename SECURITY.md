# Security policy

## Reporting a vulnerability

Please report security issues **privately**, using GitHub's
[private vulnerability reporting](https://github.com/CaullenOmdahl/subcor/security/advisories/new)
(the **Security** tab, then **Report a vulnerability**). Don't open a public issue for
anything that could be exploited.

A useful report includes:

- what an attacker can do, and what access they need for it (no auth, a tenant principal, a
  proposer, local file access, ...);
- the affected version or commit, and the backend (MLX, llama.cpp, PyTorch);
- a minimal reproduction: a request, a test case or a short script.

You can expect an acknowledgement within a week. Fixes land on `main`, and the advisory is
published once a fix is available. If you'd like credit, tell us how you want to be
credited.

## Supported versions

SUBCOR is pre-1.0. Only the latest `main` gets security fixes.

## What counts as a security issue

SUBCOR changes model behavior at runtime, so its governance checks are part of the
security boundary, not just its HTTP surface. These are in scope:

- **Authentication and tenant isolation** in `subcor serve`: reading or changing another
  tenant's trajectories, units or chats; acting as a different principal.
- **The governance chain:**
  - getting a unit to `CONFORMANT` or `ENABLED` without passing the owner's conformance
    policy;
  - self-approval (approver equal to proposer);
  - loading a payload whose SHA-256 doesn't match its manifest;
  - feeding unverified trajectories into training.
- **Unsafe file handling:** path traversal through unit IDs, tenant IDs or export paths;
  unsafe deserialization of unit bundles or GGUF files.
- **Process and network exposure:** the supervised `llama-server` subprocess, or default
  bind addresses.

These are **not** security issues on their own:

- **Seat scope.** Seat scope is a routing hint, not access control (see
  [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)).
- **Model output.** What a base model says, or a unit's quality, is not a security issue
  unless it comes from bypassing a governance check.
- **Unauthenticated deployments.** Running `subcor serve` without auth on a network you
  don't trust is unsafe. The server refuses non-loopback binds without an API key or auth
  config, unless you pass `--allow-unauthenticated`.

## Deployment notes

- Keep `subcor serve` on `127.0.0.1` and reach it through an SSH tunnel or a reverse proxy.
  If you must bind to a LAN address, set `--api-key` or `--auth-config`.
- Give proposers and approvers separate principals. The no-self-approval rule only protects
  you if they really are different people or services.
- Set the conformance policy (`ConformancePolicy`) in server-owner code. Study requests can
  tighten it but never loosen it.
