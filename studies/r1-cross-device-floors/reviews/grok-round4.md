I'll check Deviation 4 against the round-3 blockers and the current card and code, starting with the omitted review text and the live scripts.

The pasted card and scripts are the claim. I'll check the live runner, analysis, and queue for the round-3 holes Deviation 4 says it closed.

The repo is on disk. I'll compare the live queue, analysis, and tests with the round-3 holes Deviation 4 says are closed.

Deviation 4 closes both round-3 blockers and both majors on the confirmatory path.

### Round 3 partly

1. Scheduler identity — **resolved**. H1a aborts if `vllm:prompt_tokens` is missing or differs across its six runs. `vllm:prompt_tokens_cached` is a separate counter. Chunked prefill is in the engine read-back.
2. One image — **resolved**. H1a and every non-5080 H1b run abort unless `image_digest` is `vllm/vllm-openai@sha256:cb9bc64f0f038c2c2ebf13b31248fb87aca40c356c3d20f363be087cfcff1fcd`. A 5080 run on another digest is reported in `device_build_pairs`, outside the overall verdict.
4. H1 multiplicity — **resolved**. H1a raises unless each confirmatory device has exactly three clean warm transfer restarts. H1b `passed` requires two clean cold and three clean warm restarts on every required device, and a match on every emitted pair.
7. Cold/warm floors — **resolved**. The floor key is device, model, mode, arm, and item file. Warm-warm, cold-warm, cold-cold, same-process, and order stay separate, and both same-process passes are kept.
8. Screen, adapters, cost rule, logprobs, McNemar, set size, weights, novelty — **partly**. Equal-benchmark weights and McNemar are inside the floor pairs. `analyze_manifest` does not call `interval_rule`, `set_size_curve`, or `rescore_check`.
9. Queue, `limit`, zero-flip bound — **resolved**. Non-MMLU clusters are the item id. A one-item floor's cluster-bootstrap upper bound can sit above the point estimate.
11. Unscored items — **resolved**. A generated id missing from `scores.json` raises. H1a and H1b abort unless the scored ids equal the transfer-core file (2,244 ids).
12. Stack gates — **resolved**. The gate compares read-back `cudagraph_mode` and `breakable_cudagraph` to `FULL_AND_PIECEWISE`, with breakable on only for batch-invariant. The RTX 6000 queue log shows the default and batch-invariant warmups finishing clean, so that read-back matched on the pin.

### Round 3 new

Warm cache — **resolved**. One uncounted warmup per device, model, and mode is prepended and shares that warm cache directory. A warm run aborts when the start listing is empty. Cold runs use a private directory.
Aborted runs — **resolved**. The queue treats a run as done only when `aborted` is absent, moves an aborted directory aside, reruns once, and stops on a second abort. Analysis loads only clean, non-warmup metas.
Floor key and 2,244 ids — **resolved**. Matched-Rane and the full set have their own floor keys. H1a and H1b abort on any other scored id set.
Prompt tokens, graphs, digest — **resolved**. Covered by lines 1, 2, and 12 above.

no open blockers
