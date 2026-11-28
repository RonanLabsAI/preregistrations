# R1 vLLM pin decision (checked 2026-11-28, `gh api` read-only only)

## Question
Is the `docker/Dockerfile` pin `af7f9488c2210d67e1033ecdc845b087ee7fe92b` after vLLM PR #59377
(batch-invariant LoRA fixed-order shrink), and does it include #60074 (fix for #58899 restart
nondeterminism under BI)?

## Findings (with links)
- Pin commit [`af7f9488c2210d67e1033ecdc845b087ee7fe92b`](https://github.com/vllm-project/vllm/commit/af7f9488c2210d67e1033ecdc845b087ee7fe92b):
  committed 2026-09-28T05:04:11Z (`gh api repos/vllm-project/vllm/commits/af7f9488...`).
- [PR #59377](https://github.com/vllm-project/vllm/pull/59377) "[Kernel][LoRA] Add deterministic
  split-K=8 LoRA shrink for batch invariance": merged 2026-10-04T22:46:17Z, merge commit
  `138810056093301f4881050fcf2b1786939da387`.
- `gh api repos/vllm-project/vllm/compare/af7f9488...13881005...` -> `ahead_by: 397, behind_by: 0`:
  **the pin is 397 commits BEFORE #59377. It predates the card's minimum ("at or after #59377").**
- [PR #60074](https://github.com/vllm-project/vllm/pull/60074) "[Bugfix][Compile] Pin Inductor kernel
  configs under VLLM_BATCH_INVARIANT": **open as of 2026-11-28** (`state: open, merged_at: null`).
  Therefore no pin available today includes it. [Issue #58899](https://github.com/vllm-project/vllm/issues/58899)
  (greedy output changes between restarts also with `VLLM_BATCH_INVARIANT=1`; q/k-norm+RoPE autotuning)
  is likewise open. Consequence, already in the card: cold-warm BI pairs are the H1b risk; this must
  be stated in every R1 run's `run_meta.json` (r1_offline records it via `tuned_config` + log parse).
- BI support at the pin: `docs/features/batch_invariance.md` at `af7f9488` states batch invariance is
  supported on "NVIDIA GPUs with compute capability 8.0 or higher" and enabled via
  `VLLM_BATCH_INVARIANT=1`. That covers sm_100 (GB300), sm_120 (RTX PRO 6000, RTX 5080) and sm_121
  (GB10) by compute capability, but the doc makes **no per-SM bitwise-parity guarantee** and vLLM's
  reproducibility promise is same-hardware/same-version only (card, Prior work). Per-chip BI kernel
  tests cited in the card (#59972 GB300, #58718 GB10, #57456/#58495 sm_120) postdate or are not
  verifiable at this pin from GitHub code search alone; treat BI on all four chips as
  **plausible-but-unproven at any pin until the R1 smoke runs it**. This is a flagged uncertainty,
  not a reason to change the design.
- Models: the same BI doc uses `Qwen/Qwen3-1.7B` as its example, so Qwen3-1.7B in BI mode is
  documented at the pin. No source at the pin specifically tests Qwen3-4B-Instruct-2507 in BI mode
  (GitHub code/issue search found none); Qwen3 dense support in vLLM is generic. Flagged: the
  timing/logprob smokes must include one BI Qwen3-4B run per device before the grid.

## Recommendation
**Do not keep `af7f9488c` for R1.** It predates #59377, which the frozen card requires
("One vLLM pin (at or after #59377...)"). Recommend the **oldest `vllm/vllm-openai` nightly image
tag whose commit is at or after `138810056093301f4881050fcf2b1786939da387`** (the #59377 merge
commit, 2026-10-04); candidate tag form `nightly-138810056093301f4881050fcf2b1786939da387` if that
exact nightly exists, otherwise the first nightly built from main after 2026-10-04. The exact
oldest existing nightly tag could not be verified from `gh` alone (Docker Hub tag listing was not
queried: no downloads in this session) — verify the tag exists in the registry at freeze time and
record it here. `docker/Dockerfile` was **not** changed (task instruction); changing its `FROM`
line to the chosen nightly is Stephen's freeze-time action. Paper-2 comparability is weaker than
the card states once the pin moves: paper-2 runs used `af7f9488`, so R1-on-newer-pin vs paper-2
comparisons are cross-pin as well as cross-device, and must be labelled as such.

## Full-set count flag (items.py)
The card's "Full set (7,260 items, paper 2)" cannot be obtained from the current
`data/eval/*/*/dev.jsonl` files, which sum to 7,317 over the 10 core benchmarks. The only exact
7,260 in paper-2 history is the protocol set (GSM8K 659 + MATH-500 250 + IFEval 270 +
LiveCodeBench 65 + MMLU-Pro `dev_full.jsonl` 6,016; see `results/protocol/p-greedy-4k/scores.json`
and `notebook/2026-10-29.md`). `configs/r1/items/full_dev.jsonl` therefore uses that composition.
If Stephen intended all-10-benchmark dev (7,317), `items.py` must change and the card amended.
