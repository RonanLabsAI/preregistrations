# PI review, 2026-11-14 (PT)

Product: `replication-lab-series-ep2`. Runs: `rtx-base-1-a1-e810fedfc7d8a847`, `rtx-sft-1-a1-1cf9f789fb7dada8`, `rtx-cig-1-a1-307ad9685c40c649`, `rtx-compare-1-a1-d6c0f948740c0847`.
Reviewed: the outcome in `ep2-outcome-2026-11-14.md` (protocol v1 with amendments A1 and A2, https://github.com/RonanLabsAI/preregistrations/blob/6e472349b92392a6e4cd7570784acee7afbdab59/studies/replication-lab-series/ep2-protocol-v1-2026-11-14.md).
Verdict: **approved by stephen, 2026-11-14 PT**. The PI approved the result as recorded; no number changed.

## What was approved

Supported: CIG minus SFT, MATH-500 Avg@16, +23.69 points (paired bootstrap 95% CI [21.29, 26.11], n=500 problems) against the paper's claimed +26.9. Framing decision: in our run the authors' SFT recipe scored below the untrained base model (SFT 42.3 vs base 46.3 Avg@16, n=500; exploratory SFT minus base -3.99, 95% CI [-5.95, -2.04]). The paper's own MATH-500 row has SFT above base (41.5 vs 39.0); its six-benchmark aggregate has SFT below base (13.8 vs 16.5). Part of the gain over SFT is therefore recovery from a baseline that hurts. Limits: one seed, MATH-500 only, the 1.5B model only.

## Publication tier

Post (P1), with the limits stated in the post. No paper.

Rule applied (POLICY.md): every preregistered outcome is recorded here, null or not. Posts and papers cover only
results that teach something or meaningful nulls. This record does not post anything; any post text still needs
the PI's read before it goes out.
