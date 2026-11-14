# Replication lab, episode 2: CIG-SFT versus matched SFT (protocol v1)

Product: `replication-lab-series-ep2` (episode 2 of the approved work order `replication-lab-series`).
Template: `l3-replication` (standing approval by the PI, 2026-11-14 PT).
Frozen: 2026-11-14 (PT), before any training or evaluation. The training and evaluation harness is frozen by a dated amendment in this directory, with its code hash, before the smoke run; no scored run precedes that amendment.

## Claim under test

Gu, Niu, Feng, Li, Yang and Qin, "What Should SFT Learn? Conditional Information Gain for Generalizable Post-Training", arXiv:2610.12995 (v1, 2026-10-09), code https://github.com/NiuKequan/CIG-SFT-public. Table 2 (Qwen2.5-Math-1.5B, one epoch on a deduplicated 30K NuminaMath-CoT subset, Avg@16 / Pass@16, temperature 1.0, 16 samples, maximum 4,096 generated tokens):

- MATH500: standard SFT **41.5 / 81.8**; CIG-SFT **68.4 / 89.6**; base model 39.0 / 86.0.
- Six-benchmark average: SFT 13.8 / 42.3; CIG-SFT 31.5 / 58.6; base 16.5 / 51.6.
- The paper states that all SFT methods share one epoch, AdamW, cosine decay, warm-up 0.1, learning rate 5e-5, global batch 128 and maximum length 2,048, with lambda 0.3 for CIG-SFT. Table 6 reports the SFT aggregate at 13.8 to 14.3 across seeds 42 to 46, below the untrained base model's 16.5.

Why this claim: a large headline gain (+26.9 MATH500 Avg@16 points over standard SFT) from a loss reweighting with public code and data, cheap enough for one workstation GPU, with a baseline that scores below the untrained model. Whether the gain survives a matched baseline is the informative question, as in episode 1.

## Replication question

Using the authors' released code, recipe and data, does CIG-SFT beat standard SFT trained with identical settings on MATH500 Avg@16 by a margin of the claimed size?

## Public inputs (frozen)

- Code and data: `NiuKequan/CIG-SFT-public` commit `b3a2f7ffa3e8ea474158dd4950f06be29a1ad543`; recipe `example/configs/qwen2_5_math_1p5b_full_cig_sft.yaml` (git blob `3480bbd82520...`); training data `example/data/numina_cot/30k/train.parquet` (git blob `aea447216db8...`, 29,830 rows per the recipe). Full blob hashes and file SHA-256 values are recorded in the harness amendment.
- Base model: `Qwen/Qwen2.5-Math-1.5B` at the Hub revision recorded in the harness amendment.
- Evaluation set: `HuggingFaceH4/MATH-500` (500 problems) at the Hub revision recorded in the harness amendment.

## Arms, seed, units

| Unit | Arm | What it runs |
| --- | --- | --- |
| `rtx-base-1` | `base` | Evaluate the untrained base model (harness check against the paper's 39.0). |
| `rtx-sft-1` | `sft` | The authors' recipe with `use_cig_sft_loss: false`, everything else identical; then evaluate. |
| `rtx-cig-1` | `cig` | The authors' recipe unchanged (credit cache, then training); then evaluate. |
| `rtx-compare-1` | `compare` | CPU-only comparison of the three arms' per-problem results. |

Seed 42 (the paper's first seed) for training data order and initialization, and for evaluation sampling (per-problem seeds derived from 42). Episode budget does not allow more seeds; one seed per arm is a declared limitation, partly offset by the paper's own report of small seed variance.

Declared deviations: PyTorch, CUDA and flash-attention versions required by the GPU (sm_120) instead of the paper's PyTorch 2.6 and CUDA 12.4; checkpoint saves every 25 optimizer steps for preemption (training mathematics unchanged; Trainer resumes data order); one GPU with the recipe's global batch of 128 preserved through gradient accumulation.

## Evaluation

16 samples per problem, temperature 1.0, top-p 1.0, maximum 4,096 new tokens, vLLM. Prompt: the same LLaMA-Factory `qwen` chat template used in training, with the problem text as the user turn exactly as training instructions are formatted. Grading: the last `\boxed{}` answer compared to the reference answer with `math-verify` (version recorded in the amendment). The paper does not state its prompt or grader; ours is fixed in advance and applied identically to all arms.

## Metrics and uncertainty

- Primary: MATH500 Avg@16 difference, CIG minus SFT, in percentage points. 95% CI by paired bootstrap over problems (10,000 resamples, seed 42) of per-problem mean correctness.
- Secondary: Pass@16 difference (same bootstrap); each arm's Avg@16 against the paper's value for that arm; base-model Avg@16 against 39.0 as a harness check.

## Decision rule

- **Supported**: CI lower bound above 0 and point difference at least 13.45 points (half the claimed 26.9).
- **Direction only**: CI lower bound above 0 and point difference below 13.45.
- **Not supported**: CI includes 0 or lies below it.
- Baseline reproduction (secondary): our SFT arm is "faithful" if within 5 points of 41.5 and "stronger than reported" if more than 5 points above.
- Harness check: if the base model's Avg@16 is more than 10 points from 39.0, the evaluation is flagged and the comparison is reported with that flag.

## Hardware, claim, interruption

One NVIDIA RTX PRO 6000 Blackwell (96 GB) on a dedicated research worker that serves no production model, under the queue's preemptible fleet claim bound to that GPU's UUID. Training checkpoints every 25 steps (maximum lost work: 25 steps, about 20 minutes); evaluation resumes per problem; the worker stops within its release budget when the claim is revoked.

## Budget, smoke, kill

- Budget: 12 GPU-hours for the episode, including smoke.
- Smoke: 5 optimizer steps of each training arm on the frozen data, a checkpoint and resume, and a 5-problem evaluation, recorded as a smoke attempt, not a result.
- Kill: input hash mismatch; loss not finite; training runs over budget; evaluation output truncated on more than 20% of samples in any arm (reported with the result rather than silently retried); preemption more than three times in one unit. Every attempt is recorded.

## What a result can and cannot establish

It can establish whether the authors' code reproduces a gain of the claimed size over their own SFT setting on MATH500 at one seed. It cannot establish behaviour on the other five benchmarks, the 7B or code results, other seeds, or whether a better-tuned SFT baseline would close the gap.

## Outcome recording

Every run outcome, including direction-only, not-supported, killed and unrun dispositions, is recorded publicly in this repository. Posts or papers only for meaningful nulls, at the PI's chosen tier.
