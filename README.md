# Inference Engineering

Hands-on labs in LLM inference: serving, caching, quantization, throughput.
Every lab runs on free-tier GPUs, ships a notebook, and reports measured numbers —
no vibes, only benchmarks.

## Labs

| # | Lab | Result |
|---|---|---|
| 01 | [Prefix caching on vLLM (Qwen3-4B, T4)](01-prefix-cache-t4/) | 7.54× prefill speedup from KV-block reuse |
| 02 | [Quantization: AWQ-4bit vs fp16 (Qwen3-4B, T4)](02-quantization-awq/) | 3× smaller, 2.25× KV room, 0× faster |

## Method

Each lab follows the same contract:

1. Serve a real model on a real (free) GPU
2. Benchmark with confounds removed (warmup, fixed seeds, isolated variables)
3. Report cold vs warm / before vs after with identical outputs
4. State exactly how to reproduce for $0
