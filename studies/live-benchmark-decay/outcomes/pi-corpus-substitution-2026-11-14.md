# PI decision on the corpus substitution, 2026-11-14 (PT)

Product: `live-benchmark-decay`. Protocol: `protocol-v1-2026-11-14.md` (commit `58f2aa354cf6d252b5f94b143afaf4690945d4bc`), as amended by `amendment-a1-2026-11-14.md` (commit `e21101fa15381704d40972fdffa9d7ab4f01b823`).
Decision: **substitution accepted by Stephen (PI)**.

## What was accepted

The approved work order asked how fast live-web benchmarks drift, using frozen-vs-live retrieval logs. It named no public, releasable corpus. Protocol v1 answered the same decision question with public, dated live benchmarks instead. The primary analysis uses the LiveCodeBench leaderboard data pinned at `LiveCodeBench/livecodebench.github.io` commit `f186ab5041d6b768733b8ae8145151c867942134`. The secondary analysis uses LiveBench per-model judgments, also pinned and hashed in `harness/inputs.lock.json`. No retrieval logs were used. Protocol v1 said the PI could reject this substitution and that the outcome record would say so if he did. He did not reject it.

## What this does not change

- No number in `outcome-2026-11-14.md` or `outcomes/cpu-decay-1-a1/` changed. Nothing was rerun.
- The caveats recorded before PI review still apply. The percentile-bootstrap interval is biased for this statistic, and the statistic counts sampling noise as drift. Fixing either needs a new preregistered version.
- This note accepts the corpus only. Any publication tier for the outcome is a separate PI decision.

## Timing

Stephen gave the decision on his review page. It was recorded here 2026-11-14 at 16:23 PT. His exact answer time was not relayed, so this note gives the recording time.
