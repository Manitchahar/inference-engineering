# Chunked Prefill: Eliminating the ITL Freeze (Qwen3-4B-AWQ, vLLM, T4)

Serving **Qwen3-4B-AWQ** with **vLLM (V1 engine)** on a free Colab **T4 (15 GB)**.
Comparing monolithic prefill (`--no-enable-chunked-prefill`) against chunked prefill (vLLM V1 default) under mixed prefill and decode traffic.

## Result

| Probe | Chunked OFF (Monolithic) | Chunked ON (Chunked) | Impact |
|---|---|---|---|
| Worst Victim ITL Stall (Mid-stream collision) | **2,701 ms (2.7s)** | **86 ms** | **31.4x reduction in freeze** |
| P99 ITL (32 requests @ 2 req/s, `bench serve`) | **6,123 ms (6.1s)** | **1,511 ms (1.5s)** | **4.05x drop in tail stutter** |
| P99 TTFT (Time to First Token) | 15,911 ms (15.9s) | 11,417 ms (11.4s) | 4.5 seconds faster |
| Aggregate Output Throughput | 55.03 tok/s | 59.35 tok/s | +7.8% higher |

**Chunked prefill turned a 2.7-second screen freeze into an 86ms hiccup.**
P99 tail stutter dropped by 4.5 seconds under load with zero throughput penalty.

## The physical mechanism: the prefill-decode collision

In LLM serving, decode is fast and memory-bound (~14 ms/step on T4), while prefill is compute-heavy (hundreds of ms for 1k+ tokens).

Without chunked prefill, whenever an incoming prompt arrives, the scheduler halts all active decode streams to compute the entire prompt in a single monolithic forward pass. Active users watching a stream experience a complete freeze (Inter-Token Latency spike).

Chunked prefill enforces a strict token budget per forward pass. Long prompts are sliced into bite-sized chunks and co-scheduled alongside active decodes in the exact same iteration. Active users receive their next token every step without interruption.

## Why TTFT improved instead of degrading

The classic theoretical trade-off predicts chunking might slightly increase Time-To-First-Token due to chunking overhead. Under real load (2 req/s), P99 TTFT actually **dropped from 15.9s to 11.4s**. 

Why: Monolithic prefills cause severe head-of-line blocking in the HTTP queue. Breaking prefills into chunks allows new requests to be admitted earlier into the iteration schedule, clearing queue backlog faster.

## Reproduce ($0)

1. Open `chunked_prefill_qwen3_t4.ipynb` on Colab (T4 GPU runtime)
2. Run baseline server: `vllm serve Qwen/Qwen3-4B-AWQ --dtype float16 --no-enable-chunked-prefill`
3. Run optimized server: `vllm serve Qwen/Qwen3-4B-AWQ --dtype float16` (Chunked prefill is default in vLLM V1)
4. Compare victim stream ITL and `vllm bench serve` outputs to the table above

## Files

- `chunked_prefill_qwen3_t4.ipynb` — full lab notebook (server setups, streaming victim probe, native `vllm bench serve` runs)
