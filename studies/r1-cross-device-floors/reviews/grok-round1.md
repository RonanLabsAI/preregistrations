The card is falsifiable only after you narrow the primary estimand. As written, a cross-chip gap is compatible with scheduler mix, a different kernel build, and a different item protocol, so it does not answer the transfer question.

### 1. Primary question
**Blocker.** "Same config, chip A vs chip B" is not identified at concurrency 96. A faster GPU builds different batch shapes under the same arrival rate, so the cross-chip gap mixes numeric divergence with scheduler mix. `max-num-seqs 128` on a 16 GB RTX 5080 will not realize that concurrency; the engine will OOM at KV-cache profiling or silently run a smaller batch while the argv still matches. Fix: make the primary transfer arm offline `LLM.generate` or HTTP concurrency 1, with the logged realized max concurrent sequences required to match. Keep concurrency 96 as a secondary serving-realism arm on GPUs that actually hold it.

**Blocker.** One image is not one binary. The GB10 is Grace (aarch64); the Station GPUs are x86; the 5080 runs under WSL. Wheels, fused kernels, and the autotuned backend will differ. Driver, CUDA, and attention backend are "logged per chip," which permits the confound Integer Alibi already demonstrated. Fix: force one named attention backend, record wheel hashes and selected kernel, and put any pair whose backend, CUDA-graph capture, or ISA differs into a pre-declared stack stratum. State the GB10 contrast as device plus build. The memory-system story is not identified.

**Major.** One board per SKU cannot support a hardware-class law. Three restarts with a single cold start cannot power "1.5–3× on every pair," and the cold-cache claim Blagojevich actually made needs two cold instances. Fix: claim these four serial numbers. Pre-specify one primary pair.

### 2. Design
**Blocker (Rane / Q3).** The grid is a different model-mix, item set, token cap (4,096 vs their GSM8K cap), vLLM pin, GPU, and HTTP path from Rane's 1.38%. "Within 2× of 1.38%" and "why paper 2 is 4–6×" are not identified. Paper 2's 6.8–8.8% are verdict flips; do not compare them with a differently graded rate. Fix: one matched arm, offline, Qwen3-4B-Instruct-2507, their cap, their items or an item-by-item mismatch list, on the 5080 and on one Station GPU. Run that same offline arm on the GB300, because the factor grid currently sits on the RTX PRO 6000 while paper 2 was the GB300.

**Major (floors).** Restart 1 (cold) is pooled with warm restarts 2–3 into one floor, and there is no cold–cold pair. Two shuffles and one same-process pass are thinner estimators than three restarts, so H2 compares unequal pair counts. Fix: report cold–warm, warm–warm, shuffle-vs-canonical, and same-process as separate estimands. Add a second cold container on the primary chip only.

**Major (factors).** One-factor-at-a-time cannot name "the largest contributor" when concurrency, prefix cache, and chunked prefill interact. The budget arm confounds truncation with numeric flips. The prefix-cache arm on single-turn items mostly reuses the chat template, so it is a weak stand-in for Patodiya. Fix: call the factor grid a screen. Split budget flips into hit-max-tokens vs completed. Scope the cache arm to the template prefix.

**Major (adapters).** Interleaved `max-loras 3` mixes heterogeneous batches with LoRA math. Routed items are a different set from the base floor. Fix: same items for base vs adapter. Add an interleaved run that repeats one adapter, so adapter identity is separated from mixed batches. Keep this arm on one chip and out of the transfer claim.

**Major (cost).** Throughput at concurrency 32 for 10 minutes does not price the scientific runs, and it does not settle BlockLock (>2× latency on A6000/A100/H100). Batch-invariant mode often drops CUDA graphs, which confounds the kernel cost. Fix: log graph capture in both modes. Report steady-state throughput and single-request prefill and decode latency. Claim Blackwell datacenter throughput only.

**Major (measurement).** Top-20 logprobs on every token can move vLLM onto a different kernel path, so the instrument becomes part of the floor. Fix: primary runs logprobs-off unless a 100-item smoke is byte-identical with logprobs on. Collect margins on a subsample or on flipped items only.

**Minor.** Stratified 1,000-item MMLU-Pro (seed 20261128) is fine if the pooled 2,244 is secondary. Unweighted item means let that subsample dominate GSM8K, MATH-500, IFEval, and LiveCodeBench.

### 3. Statistics
**Blocker.** H1 tests six pairs × two models × graded, extracted, and text endpoints, while the writeup's confirmation rule is "every pair." Overlapping run pairs (1–2, 1–3, 2–3) do not add independent runs. The cluster bootstrap is item-sampling error conditional on these runs. Fix: one primary endpoint, graded-outcome flip; one model, Qwen3-4B; one pair; warm–warm; test the paired within-item difference. Publish the ratio and its CI only when the within-chip floor's interval lies above 0.5%. Everything else is descriptive, with no multiplicity adjustment and no confirmatory language. The shared 2,000-draw matrix is the right dependence structure for those descriptive contrasts.

**Major.** Clopper–Pearson on items treats clustered items as iid and will be tight. The largest pair is an order statistic. Fix: cluster-bootstrap upper bounds for zero-flip rates. For batch-invariant identity, one byte mismatch fails the run; quote the item-level bound as descriptive. Report the max pair without a confidence interval.

**Minor.** McNemar answers directional accuracy bias. Keep it beside the flip rate and never as the flip test. For set size, write the SE as \(\sqrt{p(1-p)/n}\). \(\sqrt{D}/n\) drops the \((1-p)\) term. H4's "within 20%" will pass whenever the estimator is binomial; demote it to a diagnostic.

### 4. Hypotheses
**Blocker (H1).** "Greater on every pair" plus "1.5–3×" has no rejection region: 1.4× and 4× both miss the band while the inequality can still hold, and a near-zero floor makes the ratio explode. Naming the 5080 and cold cache as expected risks makes a failure a confirmation. Fix: H1a, primary pair, cross-chip graded rate exceeds that chip's warm restart floor, CI on the paired difference excludes 0. H1b, batch-invariant text identity is pass/fail per pair, including the 5080 and the cold restart, with no escape hatch.

**Major (H2, H5).** Confirm H2 only if the intervals on (restart − same-process) and (reorder − restart) exclude 0; otherwise inconclusive. One same-process pair cannot establish "non-zero." H5 needs a numeric rule (byte identity for merged and single-LoRA; interleaved-mixed exceeds interleaved-same). A Granite smoke is not evidence about Qwen.

**Major (H3, H6).** Replace H3 with the matched reproduction plus a screen whose intervals separate. H6 as 5–25% confirms almost any mild regression. Fix: estimate the cost; call it closer to Rane than BlockLock only if throughput loss is under 50% on both datacenter chips.

**Minor (H4).** Keep as a check, not a hypothesis.

### 5. Missing checks under 30 GPU-h
| Add | Role | Cost |
|---|---|---|
| Matched Rane offline 4B arm, 5080 + one Station GPU | Identifies the 1.38% comparison | about 2–4 h |
| Concurrency-1 or offline transfer, 4B, warm restarts only, all four chips | Primary estimand | replace the conc-96 grid on the 5080 and GB10 |
| Second cold restart, primary chip, 4B | Tests the cache claim | about 1 run |
| Logprobs on/off smoke; CUDA-graph and backend gate | Stops instrument and stack confounds | smoke budget |
| Interleaved single-adapter control | Identifies H5 | swap, not an add |
| OSF or public gist of the freeze hash before the first run | Makes the prereg externally checkable | free |

Drop if the timing smoke exceeds 30 h, in this order: Qwen3-8B; full 7,260 on the second Station GPU (keep the full set on the GB300 for the set-size curve and paper 2); GB10 shuffle orders; the mixed-adapter arm if the same-adapter control does not fit. Do not drop the 5080's 4B offline arm. That is the consumer bridge.

### 6. Novelty
The fair headline is narrower than the card's six contributions. Cross-device transfer of a same-config floor, and LoRA identity under batch-invariant kernels, are real extensions: Rane and Blagojevich are one GPU, Zhang et al. showed H100≠A100 batch-invariant parity and stopped there, and the Cortex consult found no LoRA determinism result. Rank those two. Model size stays a covariate.

Overclaim, **major**: "what makes paper 2 4–6× Rane," "BI cost settles BlockLock," and "batch-invariant byte identity across chips" as a hardware result. Zhang already owns the qualitative non-uniformity result. BlockLock's number is a latency penalty on older architectures. DriftBench is correctly dismissed. The restart and shuffle protocol is correctly labeled infrastructure. Say Blackwell SKUs, these builds, these serving paths.

### Freeze these five, in order
1. Primary arm is scheduler-matched (offline or concurrency 1), one forced attention backend, stack-mismatched pairs stratified. Claim four devices, not chip classes.
2. One confirmatory test: Qwen3-4B, graded flip, one pre-specified pair, warm–warm, paired difference. H1b is pass/fail identity with no escape hatch. Park 1.5–3×.
3. Add the matched Rane offline arm on the 5080 and one Station GPU, plus the same offline path on the GB300. Withdraw the reproduction sentence until that arm exists.
4. Unpool cold and warm; add a second cold start on the primary chip; logprobs off on primary runs unless the smoke matches; H2 and H5 use interval rules.
5. Rebudget by dropping 8B, the second full 7,260, and GB10 shuffles before touching the 5080 4B arm or the matched transfer. Timestamp the frozen card on a public prereg before the first grid run.
