Deviation 3 says the hashed analysis is now the carded test. I am checking each round-2 blocker and major against the runner, the analysis entrypoint, and the queue, including the paths that still admit a run.

Deviation 3 closes the old pooling bug and pins block size. Two confirmatory holes remain: a cold compile is still labeled warm, and an aborted run is kept as data.

### Round 2

1. Scheduler identity — **partly**. Block size 16, bf16 KV, both caps, context length, prefix caching, and the attention backend are read back after `LLM()` and abort the run. Prompt-token totals are stored and do not fail H1a. At pin `3f1bcfd82`, `vllm:prompt_tokens` is registered before `vllm:prompt_tokens_cached`, so today's prefix happens to hit the prefill counter.
2. One image — **partly**. Stratum G is gone, the tag is the pin, and the digest is recorded. H1a `supported` ignores it, and `sha256:cb9bc64f…` is never compared.
3. Rane arm — **resolved**. Output caps stay Rane's, and each matched run records 32 vs 128 sequences, the 2048-block override, and block size 16.
4. H1 multiplicity — **partly**. H1a keeps exactly three warm Qwen3-4B transfer restarts with batch-invariant off, or it raises. H1b keeps only Qwen3-4B transfer restarts with batch-invariant on. Overall `passed` is true when every existing pair matches, including a device that has one run and therefore no within-device pair.
5. H1 rejection region — **resolved**. Support requires the excess interval above 0. One text or token-id mismatch fails that pair. A missing device cannot pass.
6. Four devices, not chip classes — **resolved**. Claims stay on these four boards and builds.
7. Cold/warm floors — **partly**. The five kinds are separate, and both same-process passes are kept. The floor key is only device, model, and mode, so matched-Rane and the 7,260-item run join the transfer warm-warm mean.
8. Screen, adapters, cost rule, logprobs off, McNemar, set size, weights, novelty — **partly**. The card still says this. `analyze_manifest` never emits the H2 rule, the set-size curve, or the rescore check.
9. Queue, `limit=100`, zero-flip bound — **partly**. Adapter, HTTP, cost, and logprob still have no runner, and `limit` is ignored. The bootstrap field is present. Non-MMLU items are one cluster per benchmark, so the upper bound equals the point estimate and is 0 when nothing flipped.
10. Public prereg — **not**. `preregUrl` is `pending`.
11. Unscored items — **partly**. A generated id missing from `scores.json` raises and is not stored as 0. An id absent from the run is dropped, and H1a can still return `supported`.
12. Stack gates — **partly**. `NVIDIA_TF32_OVERRIDE=0` is read back, the process is unbuffered, and an empty log aborts. Any log line containing "breakable" becomes breakable, and any other capture line becomes piecewise. Read-back `cudagraph_mode` is not compared. On this pin that enum is `PIECEWISE` or `FULL_AND_PIECEWISE`, while breakable is a separate flag.
13. Preemptions — **resolved**. They are reported and do not drop a run.
14. Prefix-cache hits — **resolved**. A missing or nonzero `vllm:prefix_cache_hits` aborts. That name does not collide with the external counter.
15. Smoke items — **resolved**. The 100 ids stay in the scored set and appear only in the H1b citation field.
16. Cold-warm excuse — **resolved**. A byte mismatch fails that pair. There is no exception branch.

### New

**Blocker.** The shared warm directory starts empty, and plan order makes `restart2` the run that fills it. H1a still treats that run as warm, so two of the three warm-warm pairs include a cold compile. `cache.n_entries` is recorded and ignored. Fix: one uncounted warmup per device, model, and mode; abort a warm run whose start listing is empty.

**Blocker.** `_write_aborted` writes `run_meta.json`. The queue treats that file as done, and analysis never reads `aborted`. Prefix-hit and engine-log failures happen after the gen files exist, so those outputs stay in H1a and H1b. Fix: resume only when `aborted` is absent; load only clean metas; require three clean warm runs for H1a, and two clean cold plus three clean warm runs on every device before H1b `passed` can be true.

**Major.** Key floors by arm and item file so matched-Rane and the full set cannot enter the transfer floor. Abort H1a and H1b unless the scored ids are the 2,244 transfer-core ids.

**Major.** Fail H1a when prompt-token totals differ or are missing, using the exact name `vllm:prompt_tokens`. Read back `enable_chunked_prefill`, `cudagraph_mode`, and `use_breakable_cudagraph`, and require every confirmatory digest to equal `sha256:cb9bc64f0f038c2c2ebf13b31248fb87aca40c356c3d20f363be087cfcff1fcd`.
