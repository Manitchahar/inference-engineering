# Quantization on vLLM: 3× Smaller, 2× Roomier, 0× Faster (Qwen3-4B, AWQ vs fp16, T4)

Same model (**Qwen3-4B**), same **Kaggle T4 (15 GB)**, same 3 prompts —
once in **fp16**, once as official **Qwen3-4B-AWQ** (4-bit, auto-detected).

## Result

| | fp16 | AWQ-4bit |
|---|---|---|
| Weights | 7.56 GiB | 2.5 GiB (3× smaller) |
| KV cache room | 4.0 GiB / 29k toks / 14× concurrency | 9.0 GiB / 65k toks / 32× concurrency |
| Decode | 70 tok/s | 64 tok/s (slightly *slower*) |
| Quality | coherent story attempt | repetitive "Also, mention..." loops |

**Quantization bought fit, not speed.** Same GPU, 2.25× the serving headroom,
zero TPS gain — the 4-bit outlier tax visible on screen.

## Why no speedup

This is weight-only quantization (W4A16): weights shrink, but compute still
happens in fp16 after on-the-fly dequantization. Bandwidth saved, math
unchanged — and at 4B scale on a T4, Marlin dequant overhead eats the bandwidth
win. Real per-token speedups need low-precision *compute* (W8A8), which is
hardware-gated to compute capability ≥ 8.9 (Ada/Hopper). On a T4 (7.5),
fp8 also falls back to weight-only.

## Honest caveats

- One run per model (warmed engine, single timed pass each)
- Both runs hit the 200-token cap (600 tokens total per model)
- Raw prompts, no chat template — both ramble; the comparison is relative,
  not an absolute quality score

## Reproduce ($0)

1. Open `awq_vs_fp16_qwen3-4b_t4.ipynb` on Kaggle (GPU T4 x2, Internet ON)
2. Run all cells — AWQ first, fp16 second, one cleanup cell between
3. Read weights/KV numbers from the engine console log
   (`torch.cuda` in the notebook reads 0 — vLLM allocates in a worker process)

## Files

- `awq_vs_fp16_qwen3-4b_t4.ipynb` — full lab notebook (both runs + outputs)
