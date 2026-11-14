# Modeled dollars per correct public math answer, study v2 (protocol v4)

Queue product: `cost-per-answer-shootout-v2`. Frozen prospectively on 2026-11-14 (PT), before any v2 output exists. This is a new study version. The v1 study (protocol v3) and its outcome stay on record unchanged; nothing here re-scores or replaces them.

## Why a v2

The v1 strict scorer counted an answer only if the whole reply was one number. Both models explained their working despite the instruction, so 0/32 (Qwen3-4B) and 1/32 (Qwen3-8B) answers qualified per cache mode, and cost per correct answer was undefined or unbounded. The PI accepted that outcome on 2026-11-14 as a method limitation, not a model result. v2 asks the same question with a scorer that reads the final numeric answer from the reply. The v1 failure mode was observed before this extractor was written; the extractor is frozen here, with its tests, before any v2 request is sent.

## Question and arms (unchanged from v3)

Which of two local models gives lower modeled GPU cost per correct answer under matched cache controls? Qwen/Qwen3-4B and Qwen/Qwen3-8B, exact revisions and file digests in `freeze.json`, BF16, pinned vLLM image, cache OFF and ON as fresh processes, order 4B OFF, 4B ON, 8B OFF, 8B ON. Same dedicated RTX PRO 6000 Blackwell Max-Q, context 8,192, TP1, eager mode, at most four sequences, one request at a time, temperature zero, seed 42, output cap 512, thinking disabled through the chat template. Same system prompt as v1 ("Answer the math problem. Return only the final number. /no_think"). Same two synthetic warmup requests, excluded from scoring.

Inputs (unchanged): first 32 test rows of openai/grade-school-math at the frozen revision in `sources.json`, byte-identical `gsm8k-32.jsonl` (SHA-256 670266375b15687d4f0e9c63bec1a8db562cb4a01fa1cee365573191ba0ba6ab). Expected value: the number after the benchmark's `####` delimiter. 32 attempts per condition, 128 in total.

## Primary scorer (new, frozen)

`runner.extract_final_number(reply, finish_reason)` in this directory, applied in this order, first rule that yields a number wins:

1. the last `#### N` in the reply;
2. the last `\boxed{N}`;
3. after the last "answer" or "final answer" marker (case-insensitive, optional "is", ":" or "="), the last number on that line; if that line has no number, the first number after the marker;
4. a reply that is only a number (the v1 rule);
5. otherwise the last number in the reply.

Before matching, `<think>` blocks, Markdown bold/underline markers and LaTeX `\( \) \[ \]` delimiters are removed. A number may carry a leading minus sign or dollar sign and thousands commas; decimals are allowed. A reply that hit the output cap (`finish_reason` "length") qualifies only through rules 1-3; rules 4-5 do not apply to it, because the last number in a cut-off reply is usually an intermediate step. Correct means numeric equality with the expected value within 1e-6. No number, an empty reply, or a mismatch is incorrect. There is no model-based judge and no manual re-scoring.

The extractor's behaviour is pinned by `test_runner.py` (synthetic text only, no study output). Its SHA-256 and the runner's are in `freeze.json`.

## Secondary scorer (v1, unchanged)

The v1 strict scorer (`runner.normalized_number`) is applied to the same replies and reported as strict-format compliance per condition with Wilson 95% intervals. It is not used for cost.

## Outcomes and analysis

Primary, per condition: extracted-correct answers out of 32 with Wilson 95% intervals; total modeled USD; modeled USD per extracted-correct answer with a 2,000-resample percentile bootstrap 95% interval over questions (seed 42; zero-correct resamples are infinite; an infinite bound is reported as unbounded).

Decision rule (new): for 4B versus 8B within each cache mode, and cache ON versus OFF within each model, the ratio of USD per correct answer with a paired bootstrap over the same 32 question ids (2,000 resamples, seed 42, 2.5th and 97.5th percentiles of defined draws). Draws with a zero denominator are counted and reported, not imputed; if more than 100 of 2,000 draws are undefined, no interval is reported and the contrast is "undefined". A condition is called lower only if the 95% interval excludes 1; otherwise "no detectable difference". These four contrasts are the only comparisons; no other ranking is claimed.

Also reported: extraction-rule counts, truncation counts, mean completion tokens, latency and token counters per row.

## Costing (unchanged from v3)

GPU energy is an estimate: trapezoid of board power sampled immediately before and after each request, times elapsed seconds. It excludes CPU, PSU, cooling and engine load, and is never called wall-plug energy or a meter reading. Hardware amortization is an assumption: USD 10,000 GPU capital, three-year straight line, zero residual, 50% useful utilization; electricity USD 0.20/kWh. Per attempt: elapsed_seconds * capital/(years*365.25*86400*utilization) + estimated_joules/3,600,000 * electricity_rate. Incorrect attempts count in the numerator. Engine loading time is reported as GPU-hours and excluded from steady-state answer cost.

## Transport (changed)

Requests go over loopback HTTP from the runner to the vLLM engine it launched on its claimed GPU. v1 routed requests through a research router and a controller proxy. Removing that hop removes router latency from each answer's elapsed time, so v2 per-answer costs are not directly comparable to v1's; v2 conditions are compared only with each other. No router credential is used.

## Safety, interruption and budget (unchanged rules)

Only a renewable preemptible fleet claim on the dedicated GPU permits work; the claim is checked before every request and engine start. Admit only with at least 90,000 MiB free. Research yields to owner use. GapRunner owns launch and container PID discovery; TERM checkpoints completed rows, stops only the owned engine and keeps prior evidence. Release bound five minutes; maximum lost work one in-flight request, at most three minutes; completed rows are durable. A foreign container is never stopped. No silent retries of failed units. Budget at most twelve GPU-hours including loading (v1 used about 0.4).

Smoke is separate from scored data: one disjoint synthetic question per condition ("What is 7 plus 8?"), real engine start, counters, packaging and K3 validation, under its own fleet claim. Kill on missing claim, production demand, insufficient memory, frozen-input or model drift, OOM, missing counters, scoring failure, failed cleanup or budget exhaustion. Engine-start failures, interruptions and unrun dispositions are outcomes, not nulls.

## Records and publication

Every answer, token counter, timing, power sample, extraction rule, finish reason, model and engine identity and failure disposition goes into immutable receipts in a K3 package. Human reports use PT. Every reported estimate carries its denominator or interval. The outcome is recorded publicly here, null or not. A post or paper requires a meaningful result and the PI's tier decision; this preregistration authorizes no post, paper, email or submission. No benchmark-general claim is made from 32 fixed questions.

## Execution

`python execute.py --product cost-per-answer-shootout-v2 --arm cost` under the queue-provided environment (`RL_RECEIPT_DIR`, `RL_RUN_ID`, unit identity, private runtime file). Add `--smoke` for the disjoint smoke. Fingerprint: `sha256:` + SHA-256 of the UTF-8 string `<freeze.json SHA-256>|cost-per-answer-shootout-v2|cost`. Authoritative identities: `freeze.json` in this directory (SHA-256 ed8d9101eeb2e9a981c9348e3b3ddc086276ba9b2ff9f82d573141eef9f289c2). Changes after this commit require an append-only dated amendment before further measurement.
