# SUBCOR on AMD and other non-Apple GPUs: GGUF serving with llama.cpp, PyTorch training

This guide covers the target setup for consumer cards with 12–16 GB of VRAM. SUBCOR
**serves** a GGUF base model through `llama.cpp` (`llama-server`) and **trains**
Skill Units itself on a PyTorch trainer loaded from the same GGUF file. Every trained
unit is exported as a GGUF LoRA adapter and loaded into `llama-server`.

The reference machine is an AMD Radeon RX 6800 (16 GB, RDNA2, `gfx1030`) on
CachyOS. It uses `llama.cpp` 1.0.0 (b12359) with the Vulkan backend. Greedy text and
logprobs for base and unit are bit-identical to b12110 on 0.8B and 4B
(`experiments/serving/results/regression-*`). It also uses `torch 2.14.1+rocm7.2`
and `transformers 5.23`. The same code paths work with CUDA, HIP and CPU builds of
`llama.cpp`, and with CUDA or CPU PyTorch.

## How it fits together

```text
             ┌──────────── subcor serve --backend llamacpp --gguf model.gguf ────────────┐
 clients ──► │ SubcorHTTPServer + TenantManager (tenants, router gates, governance)      │
             │        │ chat / embeddings (HTTP)                │ Study Cycle            │
             │        ▼                                          ▼                        │
             │  LlamaServerSupervisor ──owns──► llama-server    TorchKernel trainer       │
             │  (restart with --lora …)         (quantized GGUF, (dequantized fp16 copy    │
             │        ▲                          LoRAs at scale 0) of the SAME GGUF)       │
             │        └──── gguf_export: Skill Unit ──► GGUF LoRA adapter ◄──┘            │
             └───────────────────────────────────────────────────────────────────────────┘
```

- **Serving.** `llama-server` runs the quantized GGUF. Chat, `tools`/`tool_calls`,
  logprobs and SSE streaming go through `/v1/chat/completions`. Router embeddings come
  from `/v1/embeddings`.
- **Routing.** A routed request selects its unit's LoRA with the per-request
  `"lora": [{"id": N, "scale": 1.0}, …]` field. Every other adapter is sent with scale
  0. Global adapter state never changes, so concurrent requests cannot leak adapters
  into each other.
- **Training.** The first Study Cycle loads a `TorchKernel` from the served GGUF.
  `transformers` dequantizes it. The trainer uses the same training algorithm as the
  MLX backend: zero-initialised LoRA, an exact zero-perturbation check and optional
  binary STE. The trained deltas are packed as sub-1-bit Skill Units.
- **Loading a unit.** Before sandbox conformance evaluation and before promotion,
  `TenantManager` calls the kernel's `prepare_units` hook. The hook exports any missing
  unit to `<state-dir>/llamacpp/loras/<base>/<unit>-<digest>.gguf`. It then restarts
  `llama-server` once with the full adapter set, loaded with `--lora-scaled FILE:0`
  (global scale 0; the supervisor then verifies via `GET /lora-adapters` that every
  global scale is 0, because embeddings always use the global scales).
  `--lora-init-without-apply` is not used: on b12110 it still applies adapters at
  scale 1.0. `ENABLED` units are loaded the same way when the server starts.

## 1. Install llama.cpp

CachyOS or Arch, Vulkan backend (the reference setup):

```bash
sudo pacman -S llama-cpp ggml-vulkan
llama-server --list-devices      # should list e.g. "Vulkan0: AMD Radeon RX 6800"
```

ROCm/HIP is an alternative backend. Install `ggml-hip` instead of `ggml-vulkan`, or
build `llama.cpp` with `-DGGML_HIP=ON`. Any other `llama-server` works if it is recent
enough to support `--lora-scaled`, `--pooling` and the per-request `lora`
field. Pass its path with `--llama-server-bin`.

## 2. Install SUBCOR with the ROCm trainer

```bash
uv sync --python 3.13 --extra dev --extra rocm
uv run --extra rocm python -c 'import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))'
# True AMD Radeon RX 6800 ...
```

The `rocm` extra installs `torch` (ROCm 7.2 wheels), `triton-rocm`, `transformers`,
`accelerate`, `safetensors` and `gguf`. ROCm exposes AMD GPUs through the `torch.cuda`
API, so the trainer device is called `cuda`. ROCm may print harmless
`(null): No such file or directory` lines on stderr; you can ignore them.

If you only need inference against an existing `llama-server`, you can skip the
`rocm` extra. The plain `--llamacpp-url` mode needs only NumPy and the standard library.

## 3. Get a GGUF

```bash
uv run --extra rocm hf download unsloth/Qwen3.5-0.8B-GGUF Qwen3.5-0.8B-Q4_K_M.gguf
```

Keep model files out of the repository. The Hugging Face cache, `~/Models` or similar
locations work. Pass the file path to `--gguf`.

## 4. Serve (and train)

```bash
uv run --extra rocm subcor serve --backend llamacpp \
  --gguf ~/.cache/huggingface/hub/models--unsloth--Qwen3.5-0.8B-GGUF/snapshots/<rev>/Qwen3.5-0.8B-Q4_K_M.gguf \
  --port 8420 --state-dir ~/.local/share/subcor \
  --tenant company-a --tenant company-b
```

SUBCOR starts and supervises `llama-server` on `127.0.0.1:--llamacpp-port` (default
`8091`). It always passes `--embeddings --pooling mean --jinja`. The server log goes
to `<state-dir>/llamacpp/llama-server.log`. SUBCOR stops `llama-server` on Ctrl+C,
on SIGTERM (also while the model is still loading or units are being hydrated) and on
normal interpreter exit. On Linux `llama-server` is also started with
`PR_SET_PDEATHSIG=SIGTERM`, so it exits when SUBCOR is killed abnormally (SIGKILL, the
OOM killer). On other platforms an abnormal kill of SUBCOR can leave `llama-server`
running. If the port is already in use, SUBCOR refuses to start rather than adopt a
server it does not own.

If the supervised `llama-server` dies while SUBCOR runs (crash, device loss, OOM), the
next request restarts it with the last known-good adapter set, or with no adapters if
that fails, and logs a warning. `subcor serve` exits non-zero if `llama-server` is not
running once startup hydration is done.

| Flag | Default | Meaning |
|---|---|---|
| `--gguf PATH` | — | Base GGUF. Enables supervision and training. |
| `--llama-server-bin` | `llama-server` | Executable to run. |
| `--llamacpp-port` | `8091` | Loopback port of the supervised server. |
| `--n-gpu-layers` | `99` | `llama-server -ngl` (`0` = CPU only). |
| `--ctx-size` | `8192` | `llama-server -c`. |
| `--llama-server-arg ARG` | — | Extra `llama-server` argument (repeatable), e.g. `--llama-server-arg=-fa --llama-server-arg=on`. |
| `--lora-mode` | `auto` | `request` (per-request `lora` field) or `global` (`/lora-adapters` + reset). `auto` is `request` when supervised and `global` for an external server. |
| `--trainer-device` | `cuda` if available, else `cpu` | PyTorch device of the trainer. |
| `--trainer-dtype` | `float16` on GPU, `float32` on CPU | Trainer dtype. fp16 is the fast path on RDNA2; bf16 is emulated there. |
| `--trainer-layers` | last 6 layers of the GGUF | Comma-separated layer indices to train. |
| `--projections` | `mlp.down_proj` | Projections to train. They must have a GGUF LoRA mapping (see `subcor.gguf_export`). |
| `--unload-trainer-after-study` | off | Drop the trainer after each Study Cycle and return its VRAM to the driver. Without it the trainer stays resident, but PyTorch's cached activation memory is still released after training. |
| `--lora-id-map` | — | External server only: `unit_id` → LoRA id, as JSON or `unit=0,other=1`. |

**External server (inference only).** Without `--gguf`, SUBCOR connects to a running
server at `--llamacpp-url`. Routed units are mapped to that server's adapters with
`--lora-id-map` (the ids `GET /lora-adapters` reports for the `--lora` files it was started
with). A routed unit without an entry runs on the base model, and SUBCOR warns once per
unit. Study Cycles fail with an error that suggests `--gguf`. Start an external server with
`--embeddings --pooling mean`. With the default pooling (`none`), `/v1/embeddings`
returns `400 "Pooling type 'none' is not OAI compatible"`. SUBCOR then falls back to
`/embedding` and mean-pools the per-token vectors. This fallback works, but it gives
different vectors from a `--pooling mean` server. A router gate fitted against one
pooling setup is not portable to the other.

Qwen3.x chat templates turn on thinking by default. SUBCOR always sends
`chat_template_kwargs.enable_thinking` explicitly (`false` unless the caller asks for
thinking). Without it, a 64-token conformance answer is spent entirely in
`reasoning_content`.

## 5. Measured numbers (RX 6800 16 GB, Vulkan `llama-server` b12110, ROCm 7.2 trainer)

Serving (`-ngl 99 -c 8192 --embeddings --pooling mean`):

| Model (GGUF) | Decode | Prompt (736 tokens) | VRAM loaded / after generating |
|---|---|---|---|
| Qwen3.5-0.8B Q4_K_M | ~290–320 tok/s | ~7,000 tok/s | 0.7 / 1.2 GiB |
| Qwen3.5-4B Q4_K_M | ~110 tok/s | ~1,800 tok/s | 3.1 / 3.5 GiB |
| Qwen3.5-9B Q4_K_M | ~72 tok/s | ~1,050 tok/s | 5.3 / 5.8 GiB (6.5 GiB at `-c 32768`) |
| Hugging Face `transformers` eager fp16 (0.8B, for comparison) | ~31 tok/s | — | — |

Loading any LoRA adapter costs about 8–13% decode throughput, even for requests that
use it at scale 0.

Training and the full Study Cycle:

| What | 0.8B | 4B |
|---|---|---|
| Trainer load from the GGUF (dequantize to fp16) | ~4.8 s, 1.4 GiB | ~10 s, 7.9 GiB |
| Training step (6 × `down_proj`, rank 8) | ~108 ms, peak 1.6 GiB | ~220 ms, peak 8.2 GiB |
| Whole card during a Study Cycle (both processes + desktop) | 4.3 GiB | 13.0–13.4 GiB |
| Async `/study` end to end | ~50 s | ~255 s |
| Held-out NOC alerts, exact JSON (base → unit, 6 cases) | 0/6 → 3/6 | 0/6 → 6/6 |

Train/serve consistency (0.8B, same token ids into the trainer on ROCm fp16 and
`llama-server` on Vulkan): first-token top-1 agreed 10/10 prompts, top-1 agreed on 73/73
teacher-forced positions, median |logprob| difference ~0.10. A unit at scale 0 is
bit-identical to the base server; at scale 1 its output matched the trainer with the
unit mounted on 6/6 prompts.

Sub-1-bit packing is CPU work (about 4 s per 1024×3584 layer, 35 s per 2560×9216 layer).
The server does it outside the kernel lock, so chat keeps flowing while a unit is packed.

`llama.cpp` decodes about 10× faster than eager PyTorch on this card. That is why
PyTorch only trains here and never serves.

### Prompt cache and Skill Unit switches (b12359, 4B, `-c 32768`)

`llama-server` keeps each slot's processed prompt, so a turn that only appends to the
conversation reads just the new tokens. It **clears** that cache whenever the request's
LoRA set differs from the one that built it. SUBCOR routes on every turn and sends
every adapter id with scale 1 or 0. So a turn on a different unit, unit→base or
base→unit, re-reads the whole conversation.

Measured with `experiments/serving/cache_switch.py`, using synthetic units and
one-turn probes on a warm cache. Times are prompt processing:

| Conversation | Same unit (or base) | Any unit change |
|---|---|---|
| 4k tokens | 0.10 s (19 new tokens) | 2.2 s (full re-read) |
| 16k tokens | 0.11 s | 10.5 s |
| 30k tokens | 0.12 s | 25.2 s |

Prompt processing slows as the context grows: about 1,900 tok/s at 4k and 1,200 tok/s
at 30k.

With `--parallel 1` there is a second effect. The routing embedding request takes the
only slot, so even same-unit turns are re-read in full every time. That's 25 s per turn
at 30k. With the default slot count (auto, 4 here) the embedding lands in another slot,
and same-unit turns stay cached. The supervisor warns if it is given `--parallel 1`.

**Sticky routing (`subcor serve --sticky-routing`, off by default)** cuts the number of
switches. The kernel remembers which units each conversation's previous turn ran with,
keyed on the tenant's router, the seat, and the opening system and user messages. Chat
requests carry no conversation id. It keeps those units unless one of these holds:

- the conversation is shorter than `--sticky-min-tokens` (default 2048, estimated at
  4 characters per token), where a re-read is cheap;
- a previous unit no longer clears its own threshold or is no longer routable;
- the fresh decision's best margin above threshold beats the previous units' weakest
  margin by more than `--sticky-switch-margin` (default 0.05). The base counts as margin 0.

A kept decision is marked `sticky` in the `RoutingDecision`, and its scores are still
the current prompt's. The trade-off is routing freshness. A turn that should move to a
slightly better unit stays on the old one, and two conversations with an identical
opening share history. How often real units switch, and how much routing quality the
default margin costs, is unmeasured until trained gates exist.

See [ALORA-NOTE.md](ALORA-NOTE.md) for the mechanism and the options for avoiding
switch re-reads.

## 6. Why train on a dequantized copy of the served GGUF

Skill Units are low-rank deltas `ΔW = B·A` added to frozen base projections. The
trainer dequantizes the same GGUF file that `llama-server` serves, so the deltas are
fitted against the weights that are actually served. They are not fitted against a
separately downloaded fp16 checkpoint, which could drift from the quantized file. The
remaining difference is numerical: the fp16 matmuls of PyTorch versus the quantized
kernels of `llama.cpp`. This is the same gap a QLoRA adapter has against its
quantized base.

Sub-1-bit units are stored as binary low-rank factors. The exported GGUF LoRA holds
the reconstructed dense low-rank factors, so `llama-server` applies exactly the
stored sub-1-bit delta at scale 1.0. Scale 0 is the unmodified base. Router
embeddings are always computed with the global adapter scales at 0, so routing
coordinates are not affected by loaded units.

## 7. VRAM guidance for 12–16 GB cards

During a Study Cycle, both processes can be resident on the GPU at once:

- `llama-server`: quantized weights, plus KV cache for `--ctx-size`, plus loaded
  LoRAs. LoRAs are tiny.
- the PyTorch trainer: **fp16 dequantized** weights (about 2 bytes per parameter),
  plus activations and LoRA optimizer state.

Rough budgets (weights only; leave 1–2 GB for KV cache, activations and the desktop):

| Base | `llama-server` Q4_K_M | Trainer fp16 | Fits 12 GB? | Fits 16 GB? |
|---|---|---|---|---|
| 0.8B | ~0.5 GB | ~1.6 GB | yes | yes |
| 2B | ~1.3 GB | ~4 GB | yes | yes |
| 4B | ~2.5 GB | ~8 GB | tight | yes |
| 8B | ~5 GB | ~16 GB | no | no |

The 0.8B and 4B rows were measured (see §5): with both processes resident the 4B Study
Cycle peaked at ~13 GiB of 16 GiB including the desktop, so on a 12 GB card a 4B base
needs `--trainer-device cpu`. The other rows are estimates. An 8–9B model
**serves** comfortably on a 16 GB card, but its fp16 trainer alone is about 16 GB and
will not fit. Training bases larger than about 5B needs a trainer that runs on a
quantized base (QLoRA-style 4-bit). That trainer is not implemented yet and is listed
as future work.

Options when memory is tight:

- `--unload-trainer-after-study`: frees the trainer between Study Cycles. The next
  cycle reloads it from the GGUF.
- `--trainer-device cpu`: trains on CPU (slow, but no VRAM), while serving stays on
  the GPU.
- Smaller `--ctx-size`, or partial offload with `--n-gpu-layers`.
- Fewer `--trainer-layers`.

### Serving on 4–6 GB cards

Skill Units always run on the device that holds their base layer. llama.cpp allocates
each LoRA tensor in the base weight's buffer (`src/llama-adapter.cpp`), and no flag
splits them. They are only a few MB, so the base model and KV cache are what decide
whether it fits.

Measured on the RX 6800 (b12359, `-c 8192`, one synthetic unit active, VRAM over
idle, prompt of about 3.1k tokens read cold):

| Config | VRAM loaded / peak | Prompt | Decode |
|---|---|---|---|
| 0.8B, all layers | 0.7 / 1.2 GiB | 8,950 tok/s | 211 tok/s |
| 4B, all layers | 3.1 / 3.6 GiB | 1,970 tok/s | 100 tok/s |
| 4B, `-ngl 24` of 32 | 2.4 / 2.8 GiB | 1,500 tok/s | 26 tok/s |
| 4B, `-ngl 16` | 1.8 / 2.2 GiB | 1,190 tok/s | 19 tok/s |
| 4B, all layers, `--no-kv-offload` | 2.6 / 3.1 GiB | 1,660 tok/s | 20 tok/s |
| 9B, all layers | 5.3 / 5.8 GiB | 1,110 tok/s | 67 tok/s |
| 9B, `-ngl 20` of 32 | 3.5 / 4.0 GiB | 490 tok/s | 14 tok/s |
| 9B, `-ngl 12` | 2.4 / 2.9 GiB | 470 tok/s | 11 tok/s |

Partial offload costs far more than it saves. Moving 8 of 32 layers to the CPU cuts
4B decode from 100 to 26 tok/s. Keeping the KV cache on the CPU drops it to 20 tok/s
and saves only 0.45 GiB at 8k context.

On a small card, pick the largest base that fits **entirely**:
- 4 GB card: 4B, which fits with little headroom for the desktop.
- 6 GB card: 4B comfortably, or 9B only with nothing else using the GPU.

The KV cache is small for these hybrid models, because only every 4th layer uses full
attention. For the 4B it is about 32 KiB per token in f16, so 32k context adds about
1 GiB. That is an estimate from the GGUF metadata; it wasn't measured here.

Results are in `experiments/serving/results/edge-b12359.json`.

## 8. Limitations

- **Restart to add adapters.** `llama-server` loads LoRA adapters only at startup.
  Loading a new unit restarts it. Chat requests wait on the kernel lock during the
  restart, which takes seconds for small models.
- **Unloading.** A rolled-back unit becomes unroutable immediately: its gate is
  unregistered and it is never selected. Its adapter stays loaded at scale 0 until the
  next `llama-server` restart, which leaves it out. Adapters of candidates that failed
  conformance (`DORMANT`) are left out of the next restart the same way.
- **Global scales and embeddings.** Embedding endpoints ignore the per-request `lora`
  field and use the global scales. SUBCOR keeps those scales at 0. Do not change them
  from outside.
- **Unit switches re-read the conversation.** See "Prompt cache and Skill Unit
  switches" in §5. At 30k tokens on a 4B, one switch costs about 25 s.
  `--sticky-routing` reduces how often this happens but does not change its cost.
- **Routing reads only the end of a long message.** `llama-server` embeds an input in
  one physical batch (`-ub`, default 512 tokens). A longer last user message is clipped
  to its final tokens before routing, so routing only sees the end of a long paste.
  Before this fix, such messages failed the whole chat request with HTTP 500.
- **One trainer at a time.** Training runs under the kernel lock, so chat waits during
  the training phase. This matches the MLX backend.
- **Quantized-base training** (needed for bases larger than about 5B on 16 GB) is
  future work.
- **Linux only.** Only Linux with the Vulkan backend has been tested end to end.
  Windows and macOS builds of `llama-server` should work the same way, but they are
  untested.
