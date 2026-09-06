# Prefix Caching on vLLM: 7.5× Prefill Speedup (Qwen3-4B, T4)

Serving **Qwen3-4B** with **vLLM (V1 engine)** on a free Colab **T4 (15 GB)**.
8 requests sharing a ~1,200-token system prompt — the standard production pattern
(system instructions + per-request question).

## Result

| | Cold (prefix computed) | Warm (prefix reused) |
|---|---|---|
| Latency | 1.56 s | 0.21 s |
| Prefill throughput | 3,402 tok/s | 32,377 tok/s |

**7.54× latency reduction, ~10× prefill throughput. Identical outputs.**
Zero model changes. Zero quantization loss. Free speedup from KV-block reuse.

## Method (what makes the number trustworthy)

- `max_tokens=1` — times prefill only, not the model's reasoning output
- Throwaway warmup run — eats the one-time Triton JIT compile spike
- Same engine for both runs, `temperature=0.0` — no confounds
- Prefix caching is **on by default** in vLLM V1 (`enable_prefix_caching=True`);
  disable only with `--no-enable-prefix-caching`

## The production lesson

Prompt order is a memory-architecture decision. Dynamic content first
(`f"{query}\n\n{docs}"`) breaks the cache at token 0 and recomputes everything.
Static context first (`f"{docs}\n\n{query}"`) keeps the whole document cached —
the GPU only computes the new tail tokens.

## Reproduce ($0)

1. Open `prefix_cache_qwen3-4b_t4.ipynb` in Colab (T4 GPU runtime)
2. Run all cells — ~5 minutes, no setup beyond `pip install vllm`
3. Compare your COLD vs WARM lines to the table above

## Files

- `prefix_cache_qwen3-4b_t4.ipynb` — full lab notebook (serve + benchmark)
- `screenshot.png` — terminal output (add yours here)
