# Throughput Sweep: Ceiling, Knee, Cliff (Qwen3-4B-AWQ, vLLM, T4)

Same model (**Qwen3-4B-AWQ**), same **Colab T4 (15 GB)**, fixed shape
(128-token in, 64-token out) — three native probes, zero custom code.

## Result

| probe | throughput | latency price |
|---|---|---|
| Offline ceiling (`vllm bench throughput`) | 8.30 req/s, 531 out tok/s | n/a (no serving) |
| Flood, 64 concurrent (`vllm bench serve`) | 512.73 out tok/s (peak 960) | P99 TTFT **3.58 s**, P99 ITL 870 ms |
| Sweep knee (GuideLLM) | ~190 gen/s @ 3 req/s | TTFT ~128 ms, ITL ~24 ms |
| Past the knee | 3.3 req/s in | TTFT **24 s**, 187 queued |

**The engine can push ~510 tok/s, but only by making users wait seconds.
The sellable capacity is ~3 req/s at ~128 ms TTFT.** Ceiling, knee, cliff —
three probes, two tools agreeing.

## Why the gap

Offline throughput is compute: weights + KV, no queueing, no HTTP, no
scheduler. Served throughput pays for all three — plus the concurrency the
scheduler actually admits. The missing ~60% (531 → ~190 gen/s at good
latency) is the price of serving vs computing. That gap is the number behind
capacity planning, autoscaling, and GPU sizing.

## Honest caveats

- Single run per probe, warmed engine, one Colab T4
- Console TTFT/ITL from GuideLLM are means; p99s come from `bench serve`
  (P99 TTFT 3.58 s, P99 TPOT 113 ms at 64-way flood)
- Runs used the model default `max-model-len` 40960 (KV capacity 71,504 toks);
  at 192 toks/req this doesn't bind, but note it for reproduction
- HF requests were unauthenticated (rate limits didn't bite; model was cached)

## Reproduce ($0)

1. Open `throughput_sweep_qwen3-4b-awq_t4.ipynb` on Colab (T4 GPU runtime)
2. Add `HF_TOKEN` in the 🔑 Secrets tab (Qwen is gated)
3. Run top to bottom: offline ceiling → serve → GuideLLM sweep → bench cross-check
4. Kill leftover servers between runs: `!pkill -9 -f vllm` (one model per GPU)

## Files

- `throughput_sweep_qwen3-4b-awq_t4.ipynb` — full lab notebook (all probes + outputs)
- `benchmarks_guidellm_sweep.csv` — raw GuideLLM sweep (per-level TTFT/latency percentiles)
- `bench_serve_flood_64.csv` — flood cross-check as a readable table (raw `--save-result` JSON beside it)
