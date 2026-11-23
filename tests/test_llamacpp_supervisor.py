"""Tests for `subcor.backends.llamacpp_supervisor.LlamaServerSupervisor`.

A fake `llama-server` executable (a small Python script) stands in for the real binary:
it records its argv + pid to a JSONL file, waits a moment before binding (so a supervisor
that does not wait for health is caught), then serves `/health`, `/v1/models` and
`/lora-adapters` on the --host/--port it was given. Like llama.cpp b12110, adapters given
via `--lora` start with global scale 1.0 (even with `--lora-init-without-apply`), adapters
given via `--lora-scaled FNAME:S` start at S, and POST /lora-adapters updates scales
(mode "ignore_post" acknowledges but ignores it, "post_500" rejects it, "slow" never
becomes healthy within a test). No real model, no GPU, no fixed ports (ephemeral only).
"""

from __future__ import annotations

import contextlib
import inspect
import json
import os
import socket
import stat
import subprocess
import sys
import textwrap
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pytest

from subcor.backends.llamacpp_supervisor import LlamaServerError, LlamaServerSupervisor

FAKE_SERVER = r"""
import json, os, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

RECORD = __RECORD__
MODE = __MODE__

argv = sys.argv[1:]
with open(RECORD, "a", encoding="utf-8") as fh:
    fh.write(json.dumps({"pid": os.getpid(), "argv": argv}) + "\n")
print("FAKE-LLAMA-SERVER-STARTED pid=%d" % os.getpid(), flush=True)
print("fake llama-server stderr line", file=sys.stderr, flush=True)

if MODE == "crash":
    print("error: failed to load model", file=sys.stderr, flush=True)
    sys.exit(3)

def value(*names, default=None):
    for i, a in enumerate(argv):
        for n in names:
            if a == n and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(n + "="):
                return a.split("=", 1)[1]
    return default

adapters = []
for i, a in enumerate(argv):
    if a in ("--lora", "--lora-scaled") and i + 1 < len(argv):
        for item in argv[i + 1].split(","):
            if a == "--lora-scaled":
                path, _, sc = item.rpartition(":")
                adapters.append({"path": path, "scale": float(sc)})
            else:
                adapters.append({"path": item, "scale": 1.0})
for idx, ad in enumerate(adapters):
    ad["id"] = idx

host = value("--host", default="127.0.0.1")
port = int(value("--port", default="8080"))
time.sleep(60.0 if MODE == "slow" else 0.6)  # model "loading": nothing listens yet

class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        raw = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            if MODE == "unhealthy":
                self._send(503, {"error": {"code": 503, "message": "Loading model"}})
            else:
                self._send(200, {"status": "ok"})
        elif self.path == "/v1/models":
            self._send(200, {"object": "list", "data": [{"id": "fake", "object": "model"}]})
        elif self.path == "/lora-adapters":
            self._send(200, adapters)
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(n) or b"null")
        if self.path == "/lora-adapters":
            if MODE == "post_500":
                self._send(500, {"error": "lora update failed"})
                return
            if MODE != "ignore_post":
                for item in body:
                    for ad in adapters:
                        if ad["id"] == int(item["id"]):
                            ad["scale"] = float(item.get("scale", 0.0))
            self._send(200, {"success": True})
        else:
            self._send(404, {"error": "not found"})

ThreadingHTTPServer((host, port), H).serve_forever()
"""


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------


def _make_fake_binary(tmp_path: Path, *, mode: str = "ok") -> tuple[Path, Path]:
    bindir = tmp_path / "bin"
    bindir.mkdir(exist_ok=True)
    record = tmp_path / f"argv-{mode}.jsonl"
    script = bindir / f"llama-server-{mode}"
    body = FAKE_SERVER.replace("__RECORD__", repr(str(record))).replace("__MODE__", repr(mode))
    script.write_text(f"#!{sys.executable}\n{body}", encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script, record


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _records(record: Path) -> list[dict[str, Any]]:
    if not record.exists():
        return []
    return [json.loads(line) for line in record.read_text("utf-8").splitlines() if line.strip()]


def _pid_alive(pid: int) -> bool:
    """True if pid exists and is not a zombie (Linux /proc + macOS/BSD fallback)."""
    if Path("/proc").is_dir():
        try:
            stat_line = Path(f"/proc/{pid}/stat").read_text()
        except (FileNotFoundError, ProcessLookupError):
            return False
        return stat_line.rsplit(")", 1)[1].split()[0] != "Z"
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    res = subprocess.run(
        ["ps", "-o", "state=", "-p", str(pid)],
        capture_output=True,
        text=True,
        check=False,
    )
    state = res.stdout.strip()
    return bool(state) and not state.startswith("Z")


def _wait_dead(pid: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.05)
    return not _pid_alive(pid)


def _get(url: str, timeout: float = 5.0) -> tuple[int, Any]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        return exc.code, None


def _flag_value(argv: list[str], *names: str) -> str | None:
    for i, a in enumerate(argv):
        for n in names:
            if a == n and i + 1 < len(argv):
                return argv[i + 1]
            if a.startswith(n + "="):
                return a.split("=", 1)[1]
    return None


def _has_flag(argv: list[str], *names: str) -> bool:
    return any(a in names or any(a.startswith(n + "=") for n in names) for a in argv)


def _lora_paths(argv: list[str]) -> list[str]:
    """Adapter paths in llama-server load order (== llama lora id order)."""
    paths: list[str] = []
    for i, a in enumerate(argv):
        if a in ("--lora", "--lora-scaled") and i + 1 < len(argv):
            for item in argv[i + 1].split(","):
                paths.append(item.rpartition(":")[0] if a == "--lora-scaled" else item)
        elif a.startswith("--lora="):
            paths.extend(a.split("=", 1)[1].split(","))
    return paths


def _is_subsequence(needle: tuple[str, ...], hay: list[str]) -> bool:
    n = len(needle)
    return any(tuple(hay[i : i + n]) == needle for i in range(len(hay) - n + 1))


@pytest.fixture
def base_gguf(tmp_path: Path) -> Path:
    path = tmp_path / "models" / "base-Q4_K_M.gguf"
    path.parent.mkdir()
    path.write_bytes(b"GGUF\x03\x00\x00\x00fake")
    return path


@pytest.fixture
def supervisors() -> Any:
    """Track supervisors so a failing test never leaks a fake server."""
    created: list[LlamaServerSupervisor] = []
    yield created
    for sup in created:
        with contextlib.suppress(Exception):
            sup.stop()


def _make_sup(
    supervisors: list[LlamaServerSupervisor],
    tmp_path: Path,
    base_gguf: Path,
    binary: Path,
    **kwargs: Any,
) -> LlamaServerSupervisor:
    kwargs.setdefault("port", _free_port())
    kwargs.setdefault("lora_dir", tmp_path / "loras")
    sup = LlamaServerSupervisor(base_gguf=base_gguf, binary=str(binary), **kwargs)
    supervisors.append(sup)
    return sup


def _adapter_files(tmp_path: Path, names: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    d = tmp_path / "exported"
    d.mkdir(exist_ok=True)
    for n in names:
        p = d / f"{n}.gguf"
        p.write_bytes(f"GGUF-lora-{n}".encode())
        out[n] = p
    return out


# --------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------


def test_start_launches_with_required_flags_and_waits_for_health(
    tmp_path: Path, base_gguf: Path, supervisors: list[LlamaServerSupervisor]
) -> None:
    binary, record = _make_fake_binary(tmp_path)
    port = _free_port()
    sup = _make_sup(
        supervisors,
        tmp_path,
        base_gguf,
        binary,
        host="127.0.0.1",
        port=port,
        n_gpu_layers=7,
        ctx_size=1234,
        extra_args=("--threads", "3", "--fake-extra-flag"),
    )
    assert sup.base_url.rstrip("/") == f"http://127.0.0.1:{port}"

    sup.start()
    # start() must not return before the server is healthy (fake binds only after 0.6 s).
    status, body = _get(f"{sup.base_url}/health")
    assert status == 200 and body == {"status": "ok"}

    recs = _records(record)
    assert len(recs) == 1
    argv = recs[0]["argv"]
    model = _flag_value(argv, "-m", "--model")
    assert model is not None and Path(model).resolve() == base_gguf.resolve()
    assert _flag_value(argv, "--host") == "127.0.0.1"
    assert _flag_value(argv, "--port") == str(port)
    assert _flag_value(argv, "-ngl", "--n-gpu-layers", "--gpu-layers") == "7"
    assert _flag_value(argv, "-c", "--ctx-size") == "1234"
    assert _has_flag(argv, "--embeddings", "--embedding")
    assert _flag_value(argv, "--pooling") == "mean"
    assert _has_flag(argv, "--jinja")
    assert _is_subsequence(("--threads", "3", "--fake-extra-flag"), argv)
    assert _lora_paths(argv) == []  # nothing loaded on a plain start
    assert _pid_alive(recs[0]["pid"])

    sup.stop()
    assert _wait_dead(recs[0]["pid"]), "llama-server left running after stop()"
    with pytest.raises((urllib.error.URLError, ConnectionError, OSError)):
        urllib.request.urlopen(f"{sup.base_url}/health", timeout=1)
    sup.stop()  # idempotent


def test_defaults_n_gpu_layers_99_and_ctx_8192(
    tmp_path: Path, base_gguf: Path, supervisors: list[LlamaServerSupervisor]
) -> None:
    binary, record = _make_fake_binary(tmp_path)
    sup = _make_sup(supervisors, tmp_path, base_gguf, binary)
    sup.start()
    argv = _records(record)[-1]["argv"]
    assert _flag_value(argv, "-ngl", "--n-gpu-layers", "--gpu-layers") == "99"
    assert _flag_value(argv, "-c", "--ctx-size") == "8192"
    assert _flag_value(argv, "--host") == "127.0.0.1"
    sup.stop()


def test_restart_loads_adapters_maps_lora_ids_and_starts_unapplied(
    tmp_path: Path, base_gguf: Path, supervisors: list[LlamaServerSupervisor]
) -> None:
    binary, record = _make_fake_binary(tmp_path)
    sup = _make_sup(supervisors, tmp_path, base_gguf, binary)
    sup.start()
    first_pid = _records(record)[-1]["pid"]

    files = _adapter_files(tmp_path, ["u-zeta", "u-alpha", "u-mid"])
    mapping = sup.restart(dict(files))

    assert _wait_dead(first_pid), "old llama-server not stopped by restart()"
    recs = _records(record)
    assert len(recs) == 2
    new_pid, argv = recs[-1]["pid"], recs[-1]["argv"]
    assert new_pid != first_pid and _pid_alive(new_pid)
    assert _has_flag(argv, "--embeddings", "--embedding")
    assert _has_flag(argv, "--jinja")
    assert _flag_value(argv, "--pooling") == "mean"

    # Mapping covers exactly the requested units with distinct ids 0..n-1 ...
    assert set(mapping) == set(files)
    assert sorted(mapping.values()) == list(range(len(files)))
    # ... and each id is that unit's position in llama-server's load order.
    loaded = _lora_paths(argv)
    assert len(loaded) == len(files)
    for uid, lora_id in mapping.items():
        assert Path(loaded[lora_id]).read_bytes() == files[uid].read_bytes(), (uid, lora_id)

    # Server is healthy and no adapter is active by default (scale 0 for every id).
    status, _ = _get(f"{sup.base_url}/health")
    assert status == 200
    status, listed = _get(f"{sup.base_url}/lora-adapters")
    assert status == 200 and len(listed) == len(files)
    assert all(float(e["scale"]) == 0.0 for e in listed), listed

    # Restarting with a different set replaces the old server and remaps ids.
    mapping2 = sup.restart({"u-mid": files["u-mid"]})
    assert _wait_dead(new_pid)
    argv3 = _records(record)[-1]["argv"]
    assert mapping2 == {"u-mid": 0}
    assert [Path(p).read_bytes() for p in _lora_paths(argv3)] == [files["u-mid"].read_bytes()]

    # Restart with nothing -> plain server, empty mapping.
    pid3 = _records(record)[-1]["pid"]
    assert sup.restart({}) == {}
    assert _wait_dead(pid3)
    assert _lora_paths(_records(record)[-1]["argv"]) == []

    last_pid = _records(record)[-1]["pid"]
    sup.stop()
    assert _wait_dead(last_pid)
    assert all(not _pid_alive(r["pid"]) for r in _records(record))


def test_restart_before_start_starts_server(
    tmp_path: Path, base_gguf: Path, supervisors: list[LlamaServerSupervisor]
) -> None:
    binary, record = _make_fake_binary(tmp_path)
    sup = _make_sup(supervisors, tmp_path, base_gguf, binary)
    files = _adapter_files(tmp_path, ["u-one"])
    assert sup.restart(files) == {"u-one": 0}
    status, _ = _get(f"{sup.base_url}/health")
    assert status == 200
    assert len(_records(record)) == 1
    sup.stop()
    assert _wait_dead(_records(record)[-1]["pid"])


def test_start_fails_fast_when_binary_crashes(
    tmp_path: Path, base_gguf: Path, supervisors: list[LlamaServerSupervisor]
) -> None:
    binary, record = _make_fake_binary(tmp_path, mode="crash")
    sup = _make_sup(supervisors, tmp_path, base_gguf, binary)
    errors: list[BaseException] = []

    def _run() -> None:
        try:
            sup.start()
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    th = threading.Thread(target=_run, daemon=True)
    t0 = time.monotonic()
    th.start()
    th.join(timeout=30)
    assert not th.is_alive(), "start() hung although llama-server exited immediately"
    assert errors, "start() must raise when llama-server exits before becoming healthy"
    assert time.monotonic() - t0 < 30
    assert all(not _pid_alive(r["pid"]) for r in _records(record))


def test_start_times_out_when_never_healthy_and_kills_process(
    tmp_path: Path, base_gguf: Path, supervisors: list[LlamaServerSupervisor]
) -> None:
    params = inspect.signature(LlamaServerSupervisor).parameters
    timeout_kw = next(
        (
            k
            for k in ("health_timeout", "startup_timeout", "start_timeout", "timeout")
            if k in params
        ),
        None,
    )
    if timeout_kw is None:
        pytest.skip("LlamaServerSupervisor exposes no health-timeout parameter")
    binary, record = _make_fake_binary(tmp_path, mode="unhealthy")
    sup = _make_sup(supervisors, tmp_path, base_gguf, binary, **{timeout_kw: 2.0})
    t0 = time.monotonic()
    with pytest.raises(Exception):  # noqa: B017 - TimeoutError / RuntimeError both acceptable
        sup.start()
    assert time.monotonic() - t0 < 20
    recs = _records(record)
    assert recs, "binary was never launched"
    assert _wait_dead(recs[-1]["pid"]), "unhealthy llama-server left running after timeout"


def test_supervisor_writes_server_output_to_a_log_file(
    tmp_path: Path, base_gguf: Path, supervisors: list[LlamaServerSupervisor]
) -> None:
    binary, _ = _make_fake_binary(tmp_path)
    sup = _make_sup(supervisors, tmp_path, base_gguf, binary)
    sup.start()
    sup.stop()
    candidates: list[Path] = []
    for attr in ("log_path", "log_file", "logfile", "log"):
        val = getattr(sup, attr, None)
        if isinstance(val, (str, os.PathLike)):
            candidates.append(Path(val))
    candidates += [p for p in tmp_path.rglob("*") if p.is_file() and p.suffix in (".log", ".txt")]
    found = [p for p in candidates if p.is_file() and "FAKE-LLAMA-SERVER-STARTED" in p.read_text()]
    if not candidates:
        pytest.skip("supervisor exposes no log path and wrote no log under tmp_path")
    assert found, f"server stdout not captured in any log file: {candidates}"


def test_no_orphan_when_owner_process_exits_without_stop(tmp_path: Path, base_gguf: Path) -> None:
    """atexit cleanup: a Python process that starts a supervisor and exits must not leak it."""
    binary, record = _make_fake_binary(tmp_path)
    port = _free_port()
    code = textwrap.dedent(
        f"""
        from pathlib import Path
        from subcor.backends.llamacpp_supervisor import LlamaServerSupervisor
        sup = LlamaServerSupervisor(
            base_gguf=Path({str(base_gguf)!r}),
            binary={str(binary)!r},
            port={port},
            lora_dir=Path({str(tmp_path / "loras")!r}),
        )
        sup.start()
        print("STARTED", flush=True)
        # exit WITHOUT calling stop()
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0 and "STARTED" in proc.stdout, proc.stderr[-2000:]
    recs = _records(record)
    assert len(recs) == 1
    pid = recs[0]["pid"]
    try:
        assert _wait_dead(pid, timeout=10), "llama-server orphaned after owner exit"
    finally:
        if _pid_alive(pid):
            os.kill(pid, 9)


# --------------------------------------------------------------------------------------
# Adapters start unapplied: launch flag + verified global reset
# --------------------------------------------------------------------------------------


def _lora_args(argv: list[str]) -> list[tuple[str, str]]:
    return [(a, argv[i + 1]) for i, a in enumerate(argv) if a in ("--lora", "--lora-scaled")]


def test_adapters_are_launched_with_lora_scaled_zero(
    tmp_path: Path, base_gguf: Path, supervisors: list[LlamaServerSupervisor]
) -> None:
    """The launch flag alone must give scale 0: here POST /lora-adapters is ignored."""
    binary, record = _make_fake_binary(tmp_path, mode="ignore_post")
    sup = _make_sup(supervisors, tmp_path, base_gguf, binary)
    files = _adapter_files(tmp_path, ["u-a", "u-b"])
    assert sup.start(files) == {"u-a": 0, "u-b": 1}
    argv = _records(record)[-1]["argv"]
    assert _lora_args(argv) == [
        ("--lora-scaled", f"{files['u-a'].resolve()}:0"),
        ("--lora-scaled", f"{files['u-b'].resolve()}:0"),
    ]
    assert not _has_flag(argv, "--lora", "--lora-init-without-apply")
    status, listed = _get(f"{sup.base_url}/lora-adapters")
    assert status == 200 and [float(e["scale"]) for e in listed] == [0.0, 0.0]
    sup.stop()


def _colon_adapter(tmp_path: Path) -> dict[str, Path]:
    d = tmp_path / "odd:dir"
    d.mkdir(exist_ok=True)
    p = d / "u-colon.gguf"
    p.write_bytes(b"GGUF-lora-colon")
    return {"u-colon": p}


def test_colon_path_falls_back_to_lora_flag_and_verified_reset(
    tmp_path: Path, base_gguf: Path, supervisors: list[LlamaServerSupervisor]
) -> None:
    """`--lora-scaled` splits on ':', so such a path uses `--lora` (scale 1.0 at load) and
    relies on the POST reset, which start() verifies via GET /lora-adapters."""
    binary, record = _make_fake_binary(tmp_path)
    sup = _make_sup(supervisors, tmp_path, base_gguf, binary)
    files = _colon_adapter(tmp_path)
    assert sup.start(files) == {"u-colon": 0}
    argv = _records(record)[-1]["argv"]
    assert _lora_args(argv) == [("--lora", str(files["u-colon"].resolve()))]
    status, listed = _get(f"{sup.base_url}/lora-adapters")
    assert status == 200 and [float(e["scale"]) for e in listed] == [0.0]
    sup.stop()


@pytest.mark.parametrize("mode", ["ignore_post", "post_500"])
def test_start_fails_and_kills_server_when_global_scale_stays_nonzero(
    tmp_path: Path, base_gguf: Path, supervisors: list[LlamaServerSupervisor], mode: str
) -> None:
    """If the reset is ignored or rejected, a nonzero global scale would apply the unit to
    routing embeddings and unrouted chat: start()/restart() must refuse."""
    binary, record = _make_fake_binary(tmp_path, mode=mode)
    sup = _make_sup(supervisors, tmp_path, base_gguf, binary)
    files = _colon_adapter(tmp_path)
    with pytest.raises(LlamaServerError, match="nonzero global scales"):
        sup.start(files)
    assert not sup.is_running and sup.adapters == {} and sup.lora_id_map == {}
    assert _wait_dead(_records(record)[-1]["pid"]), "server with an applied adapter left running"

    sup.start()  # plain base server is fine
    with pytest.raises(LlamaServerError, match="nonzero global scales"):
        sup.restart(files)
    assert not sup.is_running
    assert all(_wait_dead(r["pid"]) for r in _records(record))


# --------------------------------------------------------------------------------------
# No orphan on abnormal owner exit (Linux PR_SET_PDEATHSIG)
# --------------------------------------------------------------------------------------


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="PR_SET_PDEATHSIG is Linux-only")
@pytest.mark.parametrize("start_in", ["main", "thread"])
def test_no_orphan_when_owner_is_sigkilled(tmp_path: Path, base_gguf: Path, start_in: str) -> None:
    """SIGKILL skips atexit: the child must still die. Starting from a short-lived thread
    (as HTTP request threads do) must not kill the child when that thread exits."""
    binary, record = _make_fake_binary(tmp_path)
    port = _free_port()
    code = textwrap.dedent(
        f"""
        import os, signal, threading, time, urllib.request
        from pathlib import Path
        from subcor.backends.llamacpp_supervisor import LlamaServerSupervisor
        sup = LlamaServerSupervisor(
            base_gguf=Path({str(base_gguf)!r}),
            binary={str(binary)!r},
            port={port},
            lora_dir=Path({str(tmp_path / "loras")!r}),
        )
        if {start_in!r} == "thread":
            t = threading.Thread(target=sup.start)
            t.start()
            t.join()
        else:
            sup.start()
        time.sleep(1.0)  # the starting thread is gone by now
        assert sup.is_running, "llama-server died with the thread that started it"
        urllib.request.urlopen(sup.base_url + "/health", timeout=5).read()
        print("ALIVE", flush=True)
        os.kill(os.getpid(), signal.SIGKILL)
        """
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=60)
    recs = _records(record)
    try:
        assert "ALIVE" in proc.stdout, proc.stderr[-2000:]
        assert proc.returncode == -9
        assert len(recs) == 1
        assert _wait_dead(recs[0]["pid"], timeout=10), "llama-server orphaned after SIGKILL"
    finally:
        for r in recs:
            if _pid_alive(r["pid"]):
                os.kill(r["pid"], 9)


@pytest.mark.parametrize(
    ("extra", "warns"),
    [
        (["--parallel", "1"], True),
        (["-np", "1"], True),
        (["--parallel=1"], True),
        (["-np", "4"], False),
        (["-np", "1", "-np", "2"], False),
        (["-fa", "on"], False),
    ],
)
def test_single_slot_warns_that_routing_evicts_the_chat_cache(
    tmp_path: Path, extra: list[str], warns: bool
) -> None:
    import warnings as _w

    with _w.catch_warnings(record=True) as caught:
        _w.simplefilter("always")
        LlamaServerSupervisor(
            base_gguf=tmp_path / "base.gguf", lora_dir=tmp_path / "loras", extra_args=extra
        )
    hits = [w for w in caught if "--parallel 1" in str(w.message)]
    assert bool(hits) is warns
