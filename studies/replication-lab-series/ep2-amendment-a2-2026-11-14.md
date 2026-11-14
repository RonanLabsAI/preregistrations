# Replication lab episode 2: amendment A2, harness corrections before scoring (2026-11-14 PT)

Amends `ep2-protocol-v1-2026-11-14.md` and `ep2-amendment-a1-2026-11-14.md`. Nothing has been scored. Everything not listed here is unchanged and binding.

## What happened since A1

A smoke of the A1 harness trained the SFT arm for 5 steps and then failed at evaluation (see change 1). Diagnostic smokes then ran on unpublished harness revisions. They covered a checkpoint resume from step 5 to step 7, a 5-problem evaluation of the SFT smoke model, and the full CIG path: credit cache over the 29,830 training rows (about 21 minutes), 5 steps and a 5-problem evaluation. The operator saw per-problem smoke correctness for those 5-to-7-step models. Smoke models are not results and nothing in the protocol was changed in response to them.

## Changes

1. **Generation cap.** `Qwen/Qwen2.5-Math-1.5B` has a 4,096-token context (`max_position_embeddings`), so a prompt plus 4,096 new tokens cannot fit. The evaluator now uses a 4,096-token model context and caps each sample at `min(4096, 4096 - prompt length)` new tokens, identically for all arms. This is the most generation the model allows, and the paper's 4,096-token maximum cannot have been realised beyond it either.
2. The compare unit reads the smoke evaluations when run in smoke mode, and each smoke uses fresh directories (`--smoke-tag`), so the comparison and packaging are smoked too.
3. Verdict wording: headings and caveat text contain no bare numbers, as the result-package validator requires.
4. Style-only lint changes (import order, one parenthesised string, an explicit comment on the grader's catch-all, which grades any parser failure as incorrect).
5. Frozen environment fingerprint becomes `rtxpro6000-sm120|image@cb9bc64f0f03|torch2.14.0+cu130|tf4.57.1|trl0.24.0|vllm0.32.0rc1.dev190+g3f1bcfd82|mathverify0.9.0|cig@b3a2f7f|code@82a46658f422`. Harness files are in `ep2-harness/` at this commit.

Measured during smoke, for planning only: about 15 seconds per optimizer step on this GPU, so a full 233-step arm takes about one hour.

The recorded smoke (all four units through this harness) and then execution follow public commit of this amendment.
