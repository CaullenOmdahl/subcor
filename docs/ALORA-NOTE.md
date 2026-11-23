# aLoRA Skill Units: research note

Status: research only, aLoRA not implemented. Sources are llama.cpp master `8d60873`
(2026-11-22) and the measurements in `experiments/serving/cache_switch.py`.

## Problem

`llama-server` clears a slot's cached prompt whenever a request's LoRA set differs from
the set that built the cache (`tools/server/server-context.cpp:1907`,
`lora_should_clear_cache` in `tools/server/server-common.cpp:305`). SUBCOR routes again on
every turn. It also sends the full adapter list with scales 1/0
(`LlamaCppServingAdapter._lora_request_field`).

As a result, any change of the routed unit forces a full re-read of the conversation:
unit A to unit B, unit to base, or base to unit. This was measured on the 0.8B model on
CPU with a 2.7k-token prompt:
- **Same unit:** `cache_n=2741`, 19 new tokens, about 0.1 s.
- **Any switch:** `cache_n=0`, 2,760 tokens re-read, about 1.8 s.

On the RX 6800 GPU with the 4B model, a switch costs 2.2 s at 4k tokens, 10.5 s at 16k
and 25 s at 30k. A same-unit turn costs about 0.1 s at any size
(`docs/AMD-GGUF.md` §5).

## What llama.cpp's aLoRA support does

An *activated* LoRA (aLoRA) carries an invocation token sequence in its GGUF metadata
(`adapter.alora.invocation_tokens`, `src/llama-arch.cpp:515`). It is read at load time
(`src/llama-adapter.cpp:223`).

When a request enables exactly one aLoRA, the server does the following:
1. It finds the **last** occurrence of the invocation sequence in the prompt
   (`server-context.cpp:1937`).
2. It processes everything before that point **with the adapter disabled**
   (`server-context.cpp:4329`, which sets the scale to 0 for the pre-invocation batch).
3. It turns the adapter back on from the invocation point onward
   (`server-context.cpp:3591`).
4. It keeps the prompt cache only up to the invocation start
   (`server-context.cpp:4088`).
5. A switch between aLoRAs, or from base to an aLoRA, does **not** clear the cache
   (`server-common.cpp:314`).

Some details matter for us:
- If the invocation sequence is missing, the aLoRA is silently disabled for that request
  (`server-context.cpp:1957`).
- More than one aLoRA in a request is an error ("Cannot run multiple aLoRAs in a single
  request", `server-context.cpp:1927`). So `top_k > 1` composition is impossible with
  aLoRA units.
- Mixing an aLoRA with a plain LoRA in the same request counts as "not all aLoRA", so the
  cache is cleared as before.
- The rule is asymmetric. aLoRA→base clears, because `next` must be all-aLoRA. Base→aLoRA
  and aLoRA→aLoRA keep the cache.

So with aLoRA units, the KV cache of the whole conversation is the frozen base's. Each
unit only changes tokens from the invocation point on, for example the final
`<|im_start|>assistant` header plus the reply. Switching units then costs a re-read of a
few tokens instead of the conversation.

## What SUBCOR would need to change

1. **Trainer** (`src/subcor/backends/torch_kernel.py`, adapter wrapper `forward` around
   line 796): apply the trainable and mounted deltas only at positions at or after the
   invocation start. Prefix positions must run on the base, or train and serve won't
   match.
   - This needs a per-batch position mask threaded into the wrapped linears.
   - Attention is the subtle part. With aLoRA, prefix K/V come from the base, but the
     suffix queries still see them. Our current units also change how the prefix is
     encoded, which aLoRA cannot do.
2. **Export** (`src/subcor/gguf_export.py:342`): write `adapter.alora.invocation_tokens`
   (a u32 array, `gguf.Keys.Adapter.ALORA_INVOCATION_TOKENS`). Choose the sequence per
   base chat template, e.g. the assistant-turn header tokens.
3. **Serving** (`LlamaCppServingAdapter._lora_request_field`): send only the active
   aLoRA with a non-zero scale. Never mix in plain LoRAs. Base-only turns should send
   **no** `lora` field at all.
   - Not verified: whether an all-zero list counts as equal to a base-built cache. Check
     with `cache_switch.py` before relying on it.
4. **Router**: aLoRA allows at most one unit per request, so `top_k` must be 1 for
   aLoRA units.

## Open questions and risks

- **Quality.** A unit that can't touch the prefix encoding may lose some of its gain.
  This is especially true for skills that depend on how a long tool output is read,
  not just how the reply is written. Measure it: train the same capability as a LoRA
  and as an aLoRA, then compare on the chat-history paired eval
  (`experiments/chat_prep/train_local.py` metrics).
- **aLoRA→base still clears the cache.** This could be avoided by never sending
  "base". Instead, route the no-unit case to a zero-delta aLoRA. That is a workaround
  and needs measuring.
- **Unproven ground.** The aLoRA code path in llama-server is newer and less used than
  plain LoRA. There's a TODO about multiple invocation strings at
  `server-context.cpp:1922`.

## Compared with sticky routing

Sticky per-conversation routing is a policy change with no training or export changes.
It removes most switches. It doesn't fix the switches that remain, and it trades some
routing accuracy for cache hits. aLoRA fixes the mechanism itself, but costs trainer
work and may cost some quality.

Sticky routing is now implemented behind `subcor serve --sticky-routing` (off by
default; see `docs/AMD-GGUF.md` §5). Prototype aLoRA only once real units exist, so the
quality question can be answered with paired evals.
