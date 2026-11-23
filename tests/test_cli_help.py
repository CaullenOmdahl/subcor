"""`subcor --help` and every subcommand's help must render (argparse treats % as a format)."""

from __future__ import annotations

import pytest

from subcor.cli import main

SUBCOMMANDS = (
    "demo",
    "mlx-intern",
    "mlx-surgery",
    "mlx-moe",
    "serve",
    "bench",
    "recompile",
    "inspect",
    "export-capheos",
)


@pytest.mark.parametrize("argv", [["--help"], *[[c, "--help"] for c in SUBCOMMANDS]])
def test_help_renders(argv: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main(argv)
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out
