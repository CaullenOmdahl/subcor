"""Supervise a local `llama-server` subprocess for the llama.cpp ServingKernel.

`LlamaServerSupervisor` lets SUBCOR own the `llama-server` process that serves a
GGUF base model, so that newly trained Skill Units (exported as GGUF LoRA adapters)
can be loaded. `llama-server` loads LoRA adapters only at startup, so adding an
adapter means restarting the server with the full adapter set.

The server is always launched with:

  - `--embeddings --pooling mean`: `POST /v1/embeddings` returns one pooled vector per
    input (the default pooling `none` is rejected by the OpenAI-compatible endpoint),
    and chat completions still work on the same server.
  - `--jinja`: the model's own chat template, so `tools` and
    `chat_template_kwargs.enable_thinking` work.
  - `--lora-scaled FNAME:0`: every adapter is loaded with global scale 0, and the
    serving adapter selects adapters per request. (`--lora-init-without-apply` is NOT
    enough: llama.cpp b12110 still applies such adapters at scale 1.0 by default.) After
    startup the global scales are also reset to 0 via `POST /lora-adapters` and verified,
    because the embedding endpoints always use the global scales.

The supervisor never adopts a server it did not start. If something already answers
on the configured port, `start()` raises. `stop()` is idempotent and is registered
with `atexit`, so a normal interpreter exit never leaves an orphan `llama-server`.
On Linux the child is also started with `PR_SET_PDEATHSIG = SIGTERM`, so it is
terminated when the SUBCOR process dies abnormally (SIGKILL, the OOM killer, or a
signal whose default action skips `atexit`). The child is always spawned from a
thread that lives as long as the process (the main thread or a dedicated spawner
thread), because the kernel delivers the death signal when the *forking thread*
exits. On other platforms only `atexit` / `stop()` clean up.
"""

from __future__ import annotations

import atexit
import contextlib
import ctypes
import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import warnings
import weakref
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future
from pathlib import Path
from typing import IO, Any

__all__ = ["LlamaServerError", "LlamaServerSupervisor"]

_PR_SET_PDEATHSIG = 1


def _pdeathsig_preexec() -> Callable[[], None] | None:
    """Linux: a `preexec_fn` that makes the child get SIGTERM when its parent dies.

    The libc `prctl` symbol is resolved here, in the parent, so the forked child only
    makes the foreign call (no imports, no allocation-heavy Python work). Returns None
    where `prctl` is unavailable (non-Linux).
    """
    if not sys.platform.startswith("linux"):
        return None
    try:
        prctl = ctypes.CDLL(None, use_errno=True).prctl
    except (OSError, AttributeError):  # pragma: no cover - exotic libc
        return None
    prctl.restype = ctypes.c_int
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    parent_pid = os.getpid()
    sigterm = int(signal.SIGTERM)

    def _preexec() -> None:
        prctl(_PR_SET_PDEATHSIG, sigterm, 0, 0, 0)
        if os.getppid() != parent_pid:  # parent died before prctl took effect
            os._exit(1)

    return _preexec


class _Spawner:
    """Spawn children from a thread that lives as long as the process.

    PR_SET_PDEATHSIG fires when the thread that forked the child exits, not the
    process. SUBCOR restarts llama-server from short-lived HTTP request threads, so
    those spawns are handed to one long-lived daemon thread instead.
    """

    _lock = threading.Lock()
    _queue: queue.SimpleQueue[tuple[Future[Any], Callable[[], Any]]] | None = None

    @classmethod
    def _loop(cls, q: queue.SimpleQueue[tuple[Future[Any], Callable[[], Any]]]) -> None:
        while True:
            fut, fn = q.get()
            if not fut.set_running_or_notify_cancel():  # pragma: no cover - never cancelled
                continue
            try:
                fut.set_result(fn())
            except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
                fut.set_exception(exc)

    @classmethod
    def run(cls, fn: Callable[[], Any]) -> Any:
        if threading.current_thread() is threading.main_thread():
            return fn()
        with cls._lock:
            if cls._queue is None:
                q: queue.SimpleQueue[tuple[Future[Any], Callable[[], Any]]] = queue.SimpleQueue()
                threading.Thread(
                    target=cls._loop, args=(q,), name="subcor-llama-spawner", daemon=True
                ).start()
                cls._queue = q
            q = cls._queue
        fut: Future[Any] = Future()
        q.put((fut, fn))
        return fut.result()


def _parallel_slots(args: Sequence[str]) -> int | None:
    """Value of `-np` / `--parallel` in a llama-server argument list (last one wins)."""
    value: int | None = None
    for i, arg in enumerate(args):
        raw: str | None = None
        if arg in ("-np", "--parallel") and i + 1 < len(args):
            raw = args[i + 1]
        elif arg.startswith(("--parallel=", "-np=")):
            raw = arg.split("=", 1)[1]
        if raw is not None:
            with contextlib.suppress(ValueError):
                value = int(raw)
    return value


class LlamaServerError(RuntimeError):
    """`llama-server` could not be started or did not become healthy."""


class LlamaServerSupervisor:
    """Own one `llama-server` subprocess serving `base_gguf` plus optional LoRA adapters."""

    def __init__(
        self,
        *,
        base_gguf: str | Path,
        binary: str = "llama-server",
        host: str = "127.0.0.1",
        port: int = 8091,
        n_gpu_layers: int = 99,
        ctx_size: int = 8192,
        extra_args: Sequence[str] = (),
        lora_dir: str | Path,
        log_path: str | Path | None = None,
        startup_timeout: float = 180.0,
        stop_timeout: float = 20.0,
        api_key: str | None = None,
    ) -> None:
        self.base_gguf = Path(base_gguf).expanduser().absolute()
        self.binary = binary
        self.host = host
        self.port = int(port)
        self.n_gpu_layers = int(n_gpu_layers)
        self.ctx_size = int(ctx_size)
        self.extra_args = tuple(str(a) for a in extra_args)
        if _parallel_slots(self.extra_args) == 1:
            warnings.warn(
                "llama-server with --parallel 1: the per-turn routing embedding request "
                "reuses the only slot and evicts the cached chat prompt, so every routed "
                "turn re-reads the whole conversation. Keep at least 2 slots (the default "
                "is auto).",
                RuntimeWarning,
                stacklevel=2,
            )
        self.lora_dir = Path(lora_dir).expanduser().resolve()
        self.log_path = (
            Path(log_path).expanduser().resolve()
            if log_path is not None
            else self.lora_dir.parent / "llama-server.log"
        )
        self.startup_timeout = float(startup_timeout)
        self.stop_timeout = float(stop_timeout)
        self.api_key = api_key

        self._lock = threading.RLock()
        self._proc: subprocess.Popen[bytes] | None = None
        self._log_fh: IO[bytes] | None = None
        self._adapters: dict[str, Path] = {}
        self._lora_ids: dict[str, int] = {}
        self_ref = weakref.ref(self)

        def _atexit_stop() -> None:
            sup = self_ref()
            if sup is not None:
                sup.stop()

        atexit.register(_atexit_stop)

    # ------------------------------------------------------------------ properties
    @property
    def base_url(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        return f"http://{host}:{self.port}"

    @property
    def pid(self) -> int | None:
        with self._lock:
            return self._proc.pid if self._proc is not None else None

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._proc is not None and self._proc.poll() is None

    @property
    def adapters(self) -> dict[str, Path]:
        """Adapters loaded in the running server (key -> GGUF LoRA path)."""
        with self._lock:
            return dict(self._adapters)

    @property
    def lora_id_map(self) -> dict[str, int]:
        """Adapter key -> `llama-server` LoRA id for the running server."""
        with self._lock:
            return dict(self._lora_ids)

    # ------------------------------------------------------------------ argv
    def _resolve_binary(self) -> str:
        candidate = Path(self.binary).expanduser()
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate.resolve())
        found = shutil.which(self.binary)
        if found is None:
            raise FileNotFoundError(
                f"llama-server binary '{self.binary}' not found. Install llama.cpp "
                "(e.g. 'sudo pacman -S llama-cpp ggml-vulkan' on Arch/CachyOS) or pass "
                "--llama-server-bin /path/to/llama-server."
            )
        return found

    def build_argv(self, adapters: Mapping[str, str | Path] | None = None) -> list[str]:
        """Return the full `llama-server` command line for the given adapter set."""
        argv = [
            self._resolve_binary(),
            "--model",
            str(self.base_gguf),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--n-gpu-layers",
            str(self.n_gpu_layers),
            "--ctx-size",
            str(self.ctx_size),
            "--embeddings",
            "--pooling",
            "mean",
            "--jinja",
            "--no-webui",
        ]
        if self.api_key:
            argv += ["--api-key", self.api_key]
        paths = [str(Path(p).expanduser().resolve()) for p in (adapters or {}).values()]
        for p in paths:
            if "," in p:
                raise ValueError(f"LoRA adapter path must not contain ',': {p}")
            if ":" in p:
                # `--lora-scaled` splits on ':'; such a path loads at scale 1.0 and relies on
                # the post-start global reset (which start() verifies).
                argv += ["--lora", p]
            else:
                argv += ["--lora-scaled", f"{p}:0"]
        argv += list(self.extra_args)
        return argv

    # ------------------------------------------------------------------ HTTP helpers
    def _request(self, method: str, path: str, payload: Any = None, timeout: float = 5.0) -> Any:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=data, headers=headers, method=method
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw) if raw.strip() else {}

    def _port_in_use(self) -> bool:
        try:
            with socket.create_connection((self.host, self.port), timeout=0.5):
                return True
        except OSError:
            return False

    def _log_tail(self, n_bytes: int = 4000) -> str:
        try:
            with self.log_path.open("rb") as fh:
                fh.seek(0, os.SEEK_END)
                size = fh.tell()
                fh.seek(max(0, size - n_bytes))
                return fh.read().decode("utf-8", errors="replace")
        except OSError:
            return ""

    def _wait_healthy(self) -> None:
        deadline = time.monotonic() + self.startup_timeout
        assert self._proc is not None
        while time.monotonic() < deadline:
            code = self._proc.poll()
            if code is not None:
                raise LlamaServerError(
                    f"llama-server exited with code {code} during startup. Log tail "
                    f"({self.log_path}):\n{self._log_tail()}"
                )
            try:
                body = self._request("GET", "/health", timeout=2.0)
                if isinstance(body, dict) and body.get("status") == "ok":
                    return
            except (urllib.error.URLError, OSError, ValueError):
                pass
            time.sleep(0.25)
        raise LlamaServerError(
            f"llama-server did not become healthy within {self.startup_timeout:.0f}s. "
            f"Log tail ({self.log_path}):\n{self._log_tail()}"
        )

    def _resolve_lora_ids(self, adapters: Mapping[str, Path]) -> dict[str, int]:
        """Map adapter keys to llama-server LoRA ids (by path via GET /lora-adapters)."""
        ordered = {key: idx for idx, key in enumerate(adapters)}
        if not adapters:
            return {}
        try:
            listed = self._request("GET", "/lora-adapters")
        except (urllib.error.URLError, OSError, ValueError):
            return ordered
        if not isinstance(listed, list):
            return ordered
        by_path: dict[str, int] = {}
        for item in listed:
            if isinstance(item, dict) and "id" in item and "path" in item:
                by_path[str(Path(str(item["path"])).resolve())] = int(item["id"])
        mapped: dict[str, int] = {}
        for key, path in adapters.items():
            lid = by_path.get(str(Path(path).resolve()))
            if lid is None:
                return ordered
            mapped[key] = lid
        return mapped

    def _zero_global_scales(self, lora_ids: Mapping[str, int]) -> None:
        """Make the global adapter scales 0 and verify it (embeddings use the global scales).

        Raises `LlamaServerError` if llama-server still reports a nonzero global scale, so
        a supervised server never serves routing embeddings through a Skill Unit.
        """
        if not lora_ids:
            return
        ids = sorted(set(lora_ids.values()))
        with contextlib.suppress(urllib.error.URLError, OSError, ValueError):
            self._request("POST", "/lora-adapters", [{"id": int(i), "scale": 0.0} for i in ids])
        try:
            listed = self._request("GET", "/lora-adapters")
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise LlamaServerError(f"Could not verify llama-server LoRA scales: {exc}") from exc
        nonzero = [
            item
            for item in (listed if isinstance(listed, list) else [])
            if isinstance(item, dict) and float(item.get("scale", 0.0) or 0.0) != 0.0
        ]
        if nonzero:
            raise LlamaServerError(
                f"llama-server LoRA adapters still have nonzero global scales: {nonzero}"
            )

    # ------------------------------------------------------------------ lifecycle
    def start(self, adapters: Mapping[str, str | Path] | None = None) -> dict[str, int]:
        """Start llama-server with `adapters` (key -> GGUF LoRA path) and wait until healthy.

        Returns the adapter key -> llama-server LoRA id map.
        """
        with self._lock:
            if self.is_running:
                raise LlamaServerError(
                    f"llama-server already running (pid {self.pid}); use restart() instead"
                )
            if not self.base_gguf.is_file():
                raise FileNotFoundError(f"Base GGUF not found: {self.base_gguf}")
            resolved = {str(k): Path(v).expanduser().resolve() for k, v in (adapters or {}).items()}
            for key, path in resolved.items():
                if not path.is_file():
                    raise FileNotFoundError(f"LoRA adapter for '{key}' not found: {path}")
            argv = self.build_argv(resolved)
            if self._port_in_use():
                raise LlamaServerError(
                    f"Port {self.host}:{self.port} is already in use; refusing to start a second "
                    "llama-server there (pass a free --llamacpp-port)."
                )
            self.lora_dir.mkdir(parents=True, exist_ok=True)
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self._log_fh = self.log_path.open("ab")
            self._log_fh.write(
                f"\n===== subcor: starting llama-server {time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
                f"===== argv: {' '.join(argv)}\n".encode()
            )
            self._log_fh.flush()
            log_fh = self._log_fh
            preexec = _pdeathsig_preexec()
            self._proc = _Spawner.run(
                lambda: subprocess.Popen(  # noqa: S603 - argv is built from trusted config
                    argv,
                    stdin=subprocess.DEVNULL,
                    stdout=log_fh,
                    stderr=subprocess.STDOUT,
                    preexec_fn=preexec,
                )
            )
            try:
                self._wait_healthy()
            except BaseException:
                self.stop()
                raise
            try:
                lora_ids = self._resolve_lora_ids(resolved)
                self._zero_global_scales(lora_ids)
            except BaseException:
                self.stop()
                raise
            self._adapters = resolved
            self._lora_ids = lora_ids
            return dict(self._lora_ids)

    def stop(self) -> None:
        """Terminate the subprocess (SIGTERM, then SIGKILL after `stop_timeout`). Idempotent."""
        with self._lock:
            proc, self._proc = self._proc, None
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=self.stop_timeout)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait(timeout=self.stop_timeout)
            elif proc is not None:
                proc.wait()
            if self._log_fh is not None:
                try:
                    self._log_fh.close()
                finally:
                    self._log_fh = None
            self._adapters = {}
            self._lora_ids = {}

    def restart(self, adapters: Mapping[str, str | Path]) -> dict[str, int]:
        """Restart llama-server with exactly `adapters` loaded (scale 0) and return the id map."""
        with self._lock:
            self.stop()
            return self.start(adapters)

    def __del__(self) -> None:
        proc = getattr(self, "_proc", None)
        if proc is not None and proc.poll() is None:
            proc.kill()

    def __enter__(self) -> LlamaServerSupervisor:
        if not self.is_running:
            self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
