# Contributing to SUBCOR

Thanks for your interest. SUBCOR is a small research-grade runtime with strict invariants,
so this page is mostly about keeping those invariants intact.

## Getting set up

You need Python 3.12 or 3.13 and [`uv`](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/CaullenOmdahl/subcor.git
cd subcor
uv sync --extra dev
uv run pytest                 # core suite: CPU only, NumPy only, under a minute
uv run subcor demo            # end-to-end proving ground in about a second
```

Optional backends:

| Backend | Install | Notes |
|---|---|---|
| Apple Silicon (MLX) | `uv sync --extra dev --extra mlx` | macOS on arm64 |
| AMD/NVIDIA/CPU training (PyTorch) | `uv sync --python 3.13 --extra dev --extra rocm` | see [`docs/AMD-GGUF.md`](docs/AMD-GGUF.md) |
| GGUF export only | `uv sync --extra dev --extra gguf` | pure Python |

The `mlx` and `rocm` extras conflict, so install one or the other. Tests for a backend you
haven't installed skip themselves.

## Before you open a pull request

Run all three of these and make sure they pass:

```bash
uv run ruff check src tests
uv run ruff format --check src tests
uv run pytest
```

There's no CI, so this local run is the gate. Please paste the `pytest` summary line into
your PR description.

## The invariants

These are the rules a change must not break. Each has tests, and a PR that weakens one
needs a very good reason and a matching update to
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

1. **The base is never written to.** Adapters mount and unmount around frozen weights.
   Mounting a zero-initialized unit changes no logit (`max diff = 0.0`), and rollback
   restores the base exactly.
2. **Only verified evidence trains.** Unverified trajectories never reach diagnosis,
   distillation or export.
3. **The owner's conformance policy can't be loosened by a proposer.** Requests may tighten
   the thresholds, the owner's anchors are always scored, and every score and threshold
   must be finite and in range.
4. **No self-approval.** `approver != proposer` for every promotion.
5. **Integrity is checked on every load.** A payload that doesn't match its manifest's
   SHA-256 fails closed.
6. **Routing falls back to the base.** If no gate clears its threshold, zero units mount.
7. **The core stays dependency-light.** `src/subcor` outside `backends/` needs only NumPy and
   the standard library (ADR-001). Heavy dependencies belong in an optional extra and a
   backend module.

## What makes a good change

- **Small and focused.** One behavior per PR. Refactors go separately from fixes.
- **Tested.** A bug fix includes a test that fails without the fix. New behavior includes
  tests for its edge cases (empty, invalid, non-finite, wrong tenant).
- **Measured, if you claim results.** Give the model, hardware, command and number of cases,
  and include failures. The reports in [`docs/`](docs/) show the expected level of detail.
  Don't train or evaluate on data you aren't allowed to share, and don't commit datasets,
  model weights or trained units: `.gitignore` already excludes `*.gguf`, `*.safetensors`,
  `*.npz` and `*.bin`.
- **Documented.** Update the README, `docs/ARCHITECTURE.md` or the relevant report when you
  change behavior people rely on.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/) style, for example
`fix(router): ...` or `feat(torch): ...`. Explain *why* in the body, not just what changed.

## Reporting bugs and ideas

- **Bugs:** open an issue with the command, the expected and actual behavior, and your
  platform and backend.
- **Security problems:** report privately, see [`SECURITY.md`](SECURITY.md).
- **Larger design changes**, such as a new adaptation mode, a new backend or a change to the
  unit format: open an issue to discuss before writing code.

## License

SUBCOR is licensed under [AGPL-3.0-or-later](LICENSE). By contributing, you agree that your
contributions are licensed under the same terms. Exported Skill Units default to
Apache-2.0, as described in the README.
