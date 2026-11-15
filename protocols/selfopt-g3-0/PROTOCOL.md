# SelfOptimize G3.0: TensorFold GB300 profiling and conditional code gates

Prospective protocol frozen 2026-11-14 PT. Standing generation template and
queueing after the champion were approved by the PI; execution additionally
requires a real profiled smoke and product admission. Every outcome, including
operational stops and unrun candidate dispositions, will be recorded here.
Posts, papers and upstream submissions remain separate PI decisions.

Question: where does the current native Zig champion spend critical-path time,
and does measured headroom justify a smallest code change with a plausible
whole-model gain of at least 1.05? G3.0 first produces a measurement package.
Candidate records are conditional; unknown bounds do not authorize execution.

## Frozen identity and workload

See binding.json for exact identities, input hashes, candidate mechanisms,
evidence, bound formulae and smallest changes. Champion source 7c1bc0b has the
same Git tree as tested source 624729eeb; served binary SHA256 is pinned.
Model: Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold, revision
078455ffe6472f9a52fbc1139f58b9db2881b25c. One GB300, sm103, 152 SMs,
driver 595.99.02. Zig 0.17.0, nvcc 13.2.86, safe optimization, ARM64,
native FP conversion OFF. Stored EXL3 experts, BF16 dense/activation/KV,
FP32 accumulation. MTP+copies loaded; requests draft OFF.

Context 40960, parallel 16, prompt cache zero, thinking OFF, temperature 0,
seed 1234. Sixteen supplied synthetic public prompts generate exactly 512
output tokens. Cache-off 8192/32768 input-token arrays generate 16 outputs.
These are distinct workloads: do not label the short-prompt decode benchmark
as decode at 8K/32K context. Long inputs characterize prefill, separately.
All output IDs must equal the frozen token references before timing is used.

Champion remeasurement: one excluded warm round, five retained complete
rounds each at C1 and C16. C1 averages 511/native decode seconds over 16 replies
per round; C16 is 8192 completion tokens/client round wall seconds. Report
means, sample CV and two-sided Student t95 over five round values, df4,
critical value 2.776445105. Matching means must remain inside
C1 [80.12335390502187,83.39369488073706] and C16
[267.3277037430555,278.23903858971084], with CV <=1.5%.
Any drift halts scoring without widening bands or substituting a workload.
These intervals describe repeats within one window, not hardware populations.

## Ordered execution and profile validity

1. Verify resident production fallbacks; reserve a preemptible GPU window.
2. Smoke using the frozen first prompt and exact champion server: five plain
   blocks, then five blocks traced with Nsight Systems CUDA graph-node tracing,
   CPU sampling/context-switch tracing disabled. Each is 512 tokens with exact
   reference IDs. Retain all attempts, including profiler failures. Compare
   mean native seconds and client seconds. Active/inactive tracing within one
   instrumented server is optional diagnostic data, not the sole overhead
   control. Smoke checkpoints each completed block, parks owned processes,
   restores the daily and times restoration before product admission.
3. Admit the product only with actual smoke evidence. Reacquire and remeasure
   the uninstrumented champion first using both complete matching cells above.
4. If smoke overhead <=10% and drift/noise gates pass, collect five paired
   traced/untraced blocks for each C1/C16 decode cell and each long-prefill
   length. Exclude one warm block per configuration; use identical inputs.
   Report profile metrics as medians and ranges over five retained blocks.
   Measured profiler overhead >10% stops G3.0 immediately; do not execute
   candidates or derive optimistic bounds from perturbed traces.
5. Timeline kernel duration is not bandwidth or occupancy. For EXL3 use Nsight
   Compute occupancy, register count, stalls, DRAM throughput and ALU counters
   on a representative served kernel, with separately measured overhead.
   Replay counters are screening diagnostics; no served speed claim uses them.
   If tooling cannot yield valid counters, ALU/HBM and occupancy are unknown.
   HBM ceiling requires a measured bandwidth floor; ALU ceiling requires an
   applicable mma.sync microbenchmark. Missing ceilings do not become bounds.

GPU-idle timeline gaps are only an upper bound on removable host bubble.
Without correlated host-work evidence, report host axis inconclusive. If
profile overhead is at least half the measured bubble, it is inconclusive.
Count neither inter-request idle nor model load as decode host headroom.

## Candidate order and separate gates

The binding contains Section 12 records for (1) GLM readback overlap,
(2) EXL3 strips ALU reduction, (3) one applicable remaining capability guard.
Their 1.0 admission sentinel denotes no measured bound yet. Before any patch
or candidate GPU execution, publish a dated amendment with current profile
receipts, measured applicable ceiling/share, computed bound >=1.05, exact
single surface, implementation identity and candidate budget. No bundling.
The already widened qmmf TMA guard and W14b's rejected EXL3 AHEAD change are
excluded; repeating the latter would be a duplicate. No eligible remaining
guard means an unrun not-applicable outcome. No valid profile means all three
remain unrun, never scored as failed scientific hypotheses.

Every executed candidate gets at least five alternating champion/candidate
pairs in one window, token equality first, with C1 and C16 reported separately.
Use paired log ratios and a two-sided t95 interval, exponentiated to ratios.
Promotion requires point >=1.05, CI lower >1.00, CV <=1.5%, and five cache-off
TTFT samples at both lengths whose empirical p95 (max at n5) is <=1.10 times
champion. Token mismatch reclassifies as token-changing with no promotion.
Kernel changes require seeded FP64 contract tests, served shapes/tile edges,
planted negative controls and byte repeatability before served comparisons.
Two consecutive executed candidate falsifier failures close an audited null.
No dominant applicable mechanism means stop for replanning, not force a patch.

## Budget, interruption and reporting

Total reservation budget 240 GPU-minutes, including load, waiting and daily
restoration; candidate sub-cap 120 minutes within that total. Check active
claim before every block; checkpoint and heartbeat after each block. Production
preempts. Stop new blocks at 230 minutes, reserving ten minutes for restore.
Hard per-request timeout 900 seconds; prompt blocks are the checkpoint unit.
Expected lost work is at most one block; the actual smoke measures that bound
and restoration, which determine the K2 interruption contract. Restore the
preserved daily container with its existing configuration, verify health and
both routed answers, then release claims. Never change the production router.

Preserve raw attempts privately with hashes and publish audited summaries and
all dispositions. Publish a dated amendment for any scientific protocol change
before the affected work; never backdate. An operational stop is not an audited
null or a promotion. Dense-model diagnostics are omitted: no dense champion or
matching references were supplied; they cannot inform this hero's bounds.
