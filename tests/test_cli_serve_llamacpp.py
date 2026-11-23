"""`subcor serve --backend llamacpp` wiring, against the fake llama-server binary.

`--gguf`: the CLI must build a supervisor with the requested flags, start llama-server,
stop it on exit and on a TenantManager failure, refuse to report "Ready" without a live
llama-server, and stop it on SIGTERM even while the model is still loading. External
mode: `--lora-id-map` reaches the adapter. No real model, no GPU, ephemeral ports.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
import time
from pathlib import Path
from typing import Any

import pytest

import subcor.server as server_mod
from subcor.cli import _parse_lora_id_map, main
from test_llamacpp_supervisor import (
    _flag_value,
    _free_port,
    _has_flag,
    _is_subsequence,
    _make_fake_binary,
    _pid_alive,
    _records,
    _wait_dead,
)


@pytest.fixture
def base_gguf(tmp_path: Path) -> Path:
    path = tmp_path / "models" / "base-Q4_K_M.gguf"
    path.parent.mkdir()
    path.write_bytes(b"GGUF\x03\x00\x00\x00fake")
    return path


@pytest.fixture
def no_serve_forever(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Return immediately from serve_forever; record the server instance."""
    served: list[Any] = []

    def _fake(self: Any, *args: Any, **kwargs: Any) -> None:
        served.append(self)

    monkeypatch.setattr(server_mod.SubcorHTTPServer, "serve_forever", _fake)
    return served


@pytest.fixture
def kernels(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Record the kernel passed to every TenantManager."""
    seen: list[Any] = []
    original = server_mod.TenantManager.__init__

    def _spy(self: Any, *args: Any, **kwargs: Any) -> None:
        seen.append(kwargs.get("kernel"))
        original(self, *args, **kwargs)

    monkeypatch.setattr(server_mod.TenantManager, "__init__", _spy)
    return seen


def _serve_argv(tmp_path: Path, base_gguf: Path, binary: Path, llama_port: int) -> list[str]:
    return [
        "serve",
        "--backend",
        "llamacpp",
        "--gguf",
        str(base_gguf),
        "--llama-server-bin",
        str(binary),
        "--llamacpp-port",
        str(llama_port),
        "--port",
        str(_free_port()),
        "--state-dir",
        str(tmp_path / "state"),
        "--n-gpu-layers",
        "0",
        "--ctx-size",
        "512",
        "--llama-server-arg=--threads",
        "--llama-server-arg=2",
    ]


def test_serve_gguf_starts_supervised_llama_server_and_stops_it_on_exit(
    tmp_path: Path, base_gguf: Path, no_serve_forever: list[Any], kernels: list[Any]
) -> None:
    binary, record = _make_fake_binary(tmp_path)
    port = _free_port()
    assert main(_serve_argv(tmp_path, base_gguf, binary, port)) == 0
    assert len(no_serve_forever) == 1

    recs = _records(record)
    assert len(recs) == 1, "llama-server must be started exactly once"
    argv = recs[0]["argv"]
    assert Path(_flag_value(argv, "--model") or "").resolve() == base_gguf.resolve()
    assert _flag_value(argv, "--host") == "127.0.0.1"
    assert _flag_value(argv, "--port") == str(port)
    assert _flag_value(argv, "--n-gpu-layers") == "0"
    assert _flag_value(argv, "--ctx-size") == "512"
    assert _flag_value(argv, "--pooling") == "mean"
    assert _has_flag(argv, "--embeddings") and _has_flag(argv, "--jinja")
    assert _is_subsequence(("--threads", "2"), argv)
    assert _wait_dead(recs[0]["pid"]), "llama-server left running after serve returned"

    (kernel,) = kernels
    assert kernel.supervisor is not None and kernel.lora_mode == "request"
    assert kernel.base_gguf == base_gguf.absolute()
    assert kernel.lora_dir == (tmp_path / "state" / "llamacpp" / "loras").resolve()


def test_serve_gguf_stops_llama_server_when_tenant_manager_fails(
    tmp_path: Path, base_gguf: Path, no_serve_forever: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    binary, record = _make_fake_binary(tmp_path)

    def _boom(self: Any, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("hydration exploded")

    monkeypatch.setattr(server_mod.TenantManager, "__init__", _boom)
    with pytest.raises(RuntimeError, match="hydration exploded"):
        main(_serve_argv(tmp_path, base_gguf, binary, _free_port()))
    recs = _records(record)
    assert len(recs) == 1 and _wait_dead(recs[0]["pid"])
    assert no_serve_forever == []


def test_serve_gguf_exits_nonzero_when_llama_server_is_down_after_hydration(
    tmp_path: Path, base_gguf: Path, no_serve_forever: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    binary, record = _make_fake_binary(tmp_path)
    original = server_mod.TenantManager.__init__

    def _crash_after(self: Any, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        pid = kwargs["kernel"].supervisor.pid
        os.kill(pid, signal.SIGKILL)  # llama-server dies during startup
        assert _wait_dead(pid)
        deadline = time.monotonic() + 5
        while kwargs["kernel"].supervisor.is_running and time.monotonic() < deadline:
            time.sleep(0.05)

    monkeypatch.setattr(server_mod.TenantManager, "__init__", _crash_after)
    with pytest.raises(SystemExit) as excinfo:
        main(_serve_argv(tmp_path, base_gguf, binary, _free_port()))
    assert "not running" in str(excinfo.value.code)
    assert no_serve_forever == []  # never reported Ready


def test_sigterm_while_llama_server_loads_stops_it(tmp_path: Path, base_gguf: Path) -> None:
    """SIGTERM during supervisor.start() (model still loading) is handled, not fatal."""
    binary, record = _make_fake_binary(tmp_path, mode="slow")
    argv = _serve_argv(tmp_path, base_gguf, binary, _free_port())
    code = textwrap.dedent(
        f"""
        import sys
        from subcor.cli import main
        sys.exit(main({argv!r}))
        """
    )
    proc = subprocess.Popen(
        [sys.executable, "-c", code],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        deadline = time.monotonic() + 30
        while not _records(record) and time.monotonic() < deadline:
            time.sleep(0.05)
        recs = _records(record)
        assert recs, "fake llama-server never launched"
        proc.send_signal(signal.SIGTERM)
        out, _ = proc.communicate(timeout=30)
        assert proc.returncode == 130, out[-2000:]
        assert "Interrupted" in out
        assert _wait_dead(recs[0]["pid"]), "llama-server left running after SIGTERM"
    finally:
        if proc.poll() is None:
            proc.kill()
        for r in _records(record):
            if _pid_alive(r["pid"]):
                os.kill(r["pid"], 9)


def test_external_mode_passes_lora_id_map(
    tmp_path: Path, no_serve_forever: list[Any], kernels: list[Any]
) -> None:
    common = [
        "serve",
        "--backend",
        "llamacpp",
        "--llamacpp-url",
        "http://127.0.0.1:9",
        "--port",
        str(_free_port()),
        "--state-dir",
        str(tmp_path / "state"),
    ]
    assert main([*common, "--lora-id-map", "ops-v1=0, fin-v2=2"]) == 0
    assert main([*common, "--lora-id-map", '{"ops-v1": 1}']) == 0
    assert main(common) == 0
    assert [k.lora_id_map for k in kernels] == [{"ops-v1": 0, "fin-v2": 2}, {"ops-v1": 1}, {}]
    assert all(k.supervisor is None and k.lora_mode == "global" for k in kernels)


@pytest.mark.parametrize("bad", ["ops", "ops=x", "ops=-1", "=3", "[1]", '{"a": true}'])
def test_parse_lora_id_map_rejects_bad_specs(bad: str) -> None:
    with pytest.raises(SystemExit, match="--lora-id-map"):
        _parse_lora_id_map(bad)


def test_lora_id_map_is_rejected_with_gguf(tmp_path: Path, base_gguf: Path) -> None:
    binary, record = _make_fake_binary(tmp_path)
    argv = [*_serve_argv(tmp_path, base_gguf, binary, _free_port()), "--lora-id-map", "a=0"]
    with pytest.raises(SystemExit, match="external server"):
        main(argv)
    assert _records(record) == []  # rejected before llama-server was started


def test_sticky_routing_is_off_by_default_and_configurable(
    tmp_path: Path, no_serve_forever: list[Any], kernels: list[Any]
) -> None:
    common = [
        "serve",
        "--backend",
        "llamacpp",
        "--llamacpp-url",
        "http://127.0.0.1:9",
        "--port",
        str(_free_port()),
        "--state-dir",
        str(tmp_path / "state"),
    ]
    assert main(common) == 0
    assert main([*common, "--sticky-routing", "--sticky-switch-margin", "0.1"]) == 0
    assert kernels[0].sticky_routing is None
    policy = kernels[1].sticky_routing
    assert policy is not None
    assert (policy.switch_margin, policy.min_context_tokens) == (0.1, 2048)
