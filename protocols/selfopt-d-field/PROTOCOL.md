# Study D: the DFlash2 champion in the field

Prospective protocol. Times are PT. The PI approved running parts H, X and W on 2026-11-16 ("do 1-4 now").

- **Engine:** TensorFold `7c1bc0be`, the frozen champion binary (server SHA-256 `1711a2c9...9c92`). No engine source changes.
- **Champion:** study C arm D4b. That is the champion recipe plus `--drafter incoai/GLM-5.3-Flash-DFlash2@bf582e4e --drafter-bits 0`, with requests sending `draft=true`. Evidence: `outcomes/selfopt-c-serving` at `f6ad955c`.
- **Model:** GLM-5.3-Flash EXL3 4 bpw, revision `078455ff`.

Three questions, each answered whichever way the data falls:

- **H (head-to-head, LEFT GB300).** Is the TensorFold champion faster than our production daily on the same GPU, and is its answer quality equal?
  - The daily is SGLang with NVFP4 weights and the same DFlash2 drafter, block size 7.
- **X (replication, RIGHT GB300).** Does the study C result (D4b against MTP drafts-on) hold on the second GB300? Are token IDs identical across the two devices?
- **W (realistic workloads, LEFT GB300, TensorFold only).**
  - Is drafting lossless at 8K and 30K-token prompts?
  - What does the drafter do at long context and at temperature 0.7?
  - What does a prompt cache do for repeated long prefixes?
  - Can the five prefill settings that study C left unrun (S2-S6) speed up long prompts?

## Workloads

- **Short corpus:** study C's 16 public prompts, 512 tokens each, temperature 0, seed 1234, thinking off. The token reference is study C's `TOKEN-REFERENCE.json`.
- **Long prompts:** `corpus-d.json`, built from openai/gsm8k (MIT) train rows. There are 4 prompts of about 7.9K tokens (L8K) and 4 of about 30K tokens (L30K). Each is worked examples followed by one new problem. The tokenizer is shared by both engines, SHA-256 `19e77364...a82d`.
- **Quality set:** the first 500 GSM8K test problems. They are zero-shot, thinking off, temperature 0, at most 1,024 tokens, with 16 in flight. The answer is the number after `####`, else the last number in the reply. Exact match is checked after removing commas.
- **Nonce rule:** in H, every long prompt starts with a fresh random session line, so neither engine can reuse a cached prefix. The SGLang copy also gets `/flush_cache` before every round.
- **Client:** `harness_d.py`, on the Station, posting to loopback. Requests carry `chat_template_kwargs.enable_thinking=false`.
  - TensorFold requests add `draft` and `return_token_ids`.
  - SGLang speed requests add `ignore_eos`, so every reply is 512 tokens.

## Part H: head-to-head (LEFT)

- **Engines:**
  - **T:** the TensorFold champion.
  - **S:** a private copy of the preserved daily, created with `docker run` from the daily's own image, args, mounts, env and device. Only the name, port (8011) and bind (loopback) differ. It is never routed.
- **Loads:** T1, S1, T2, S2, in that order.
  - Each load runs, in order:
    - one warm round plus 5 retained rounds at each of C1, C4 and C16 on the short corpus;
    - one warm round plus 3 rounds each of `prefill` (max 1 token) and `long256` (256 tokens) on L8K and L30K;
    - GSM8K, on T1 and S1 only.
  - T's short-corpus token IDs must equal the reference. A mismatch halts the study.
- **Metrics:**
  - **Speed:** completion tokens / round wall seconds, at every cell.
  - **Prefill:** mean prompt tokens / client seconds over the 4 prompts.
  - **long256:** mean completion tokens / client seconds, including prefill.
- **Statistics:**
  - Speed: T/S paired log ratio, pairs (T1 round i, S1 round i) and (T2 round i, S2 round i). That is 10 pairs, t95 df 9.
  - Prefill and long256: 6 pairs, df 5.
  - Quality:
    - each engine's accuracy, with a Wilson 95% interval;
    - the paired difference (T − S) with its 95% interval;
    - an exact two-sided McNemar test.
- **Reporting rule:**
  - At each cell, T is "faster" if the t95 lower bound is above 1, "slower" if the upper bound is below 1, else "no difference".
  - Quality is "parity" if the whole 95% interval of the difference is within ±3 points. Otherwise the better engine is named, or "inconclusive" if the interval spans more than ±3 points without excluding 0.
- No admission bar applies: H informs a production decision that stays with the PI.
- **If S cannot start twice,** H is unrun (two failed starts) and W proceeds.

## Part X: replication (RIGHT)

- **Driver:** `run_loads_x.py`. It is study C's driver with RIGHT constants only. Study C's `harness.py` and `analyze.py` are bound at `191b7a48` and used unchanged.
- **Loads:**
  - **X-A1:** MTP drafts-on A and drafts-off R, alternating, one warm plus 5 rounds at C1, C4 and C16.
  - **X-D4b-1, X-D4b-2:** D4b, drafted, 5 rounds.
  - **X-A2:** A, 5 rounds.
- **Cross-device check:**
  - R's IDs must equal the LEFT reference. A mismatch ends part X and is reported as the finding: "token IDs differ across our two GB300s".
  - A's and D4b's IDs must equal it too.
- **Replication rule:** the paired D4b/A ratio, pairs as in study C phase 2 (10 pairs, df 9), has a t95 lower bound above 1.05 at all three cells. Both A loads must stay inside X-A1's ±2% bands with CV at most 1.5%.
- **Frozen bands:**
  - RIGHT: freeze D4b's bands from X-D4b-1 and X-D4b-2 (mean ±2%).
  - LEFT: frozen now from study C's D4b loads, 10 rounds, metric as in study C:
    - C1: 140.243 [137.438, 143.048]
    - C4: 238.398 [233.630, 243.166]
    - C16: 331.729 [325.094, 338.363]

## Part W: realistic workloads (LEFT, TensorFold only)

- **W-R** (MTP loaded, `draft=false`):
  - long256 on L8K and L30K, 2 rounds plus warm.
  - All rounds must agree. Their IDs become the long-prompt references (`ref-L8K.json`, `ref-L30K.json`).
  - Disagreement between rounds halts W.
- **W-A** (MTP drafts-on) and **W-B** (D4b):
  - long256 on L8K and L30K, 3 rounds. IDs must equal those references. A mismatch fails that engine's long-context lossless claim and is reported.
  - temp: short corpus at temperature 0.7, top_p 0.95, C1 and C16, 3 rounds. This is speed only; sampled outputs are not compared.
  - W-B adds prefill on L30K (3 rounds) and mixed (3 rounds). Mixed means one client sends the 4 L30K prompts for 64 tokens while three clients send short prompts 4-15 for 512 tokens. The short replies' IDs must equal the reference.
- **Long-context decode:** native (completion − 1) / `tensorfold.seconds`, D4b/A, descriptive.
- **Temperature:** D4b/A, descriptive.
- **W-C** (D4b, `--prompt-cache-gib 8`): cache task, 3 rounds. It sends 4 L30K requests sharing one context with different final questions. Reported as cold / warm-mean prefill seconds, descriptive.
- **Prefill screen** (each arm on top of D4b, one load, prefill L30K and mixed, 3 rounds):

  | Arm | Setting |
  |---|---|
  | S2 | `--decode-share 0.5` |
  | S3 | `TENSORFOLD_PREFILL_POLICY=fcfs` |
  | S4 | `TENSORFOLD_PREFILL_ORDER=srpt` |
  | S5 | `TF_GLM_PROMPT_STEPS=0` |
  | S6 | `TF_GLM_CHUNK_SOLO=4096` |

  - The screen ratio is the arm's mean against W-B on L30K prefill and on mixed long_tok_s. Mixed short_tok_s is a guard.
  - An arm advances if its IDs pass, its prefill or mixed-long ratio is at least 1.03, and its mixed-short ratio is at least 0.97.
  - At most 2 arms advance, ranked by their larger gain ratio.
- **Confirm:**
  - Loads W-Bc1, WC-arm-1, WC-arm-2, W-Bc2, sharing the middle base load with the next arm. Each is one warm plus 5 rounds of prefill L30K and mixed. That gives 10 pairs, df 9.
  - **Admission:**
    - all IDs pass;
    - the t95 lower bound is above 1.05 on prefill or on mixed long_tok_s;
    - the mixed short_tok_s lower bound is at least 0.97.
  - An admitted arm becomes a long-prompt champion candidate. The PI decides.

## Budget, order and stops

- **LEFT:** one claim window for H then W, capped at **5.5 GPU-hours** from claim to release, including restore.
- **RIGHT:** a separate window afterwards, never overlapping LEFT, capped at **2.0 GPU-hours**.
- **Budget rule:** before any load, if elapsed time plus that load's estimate plus a 0.25 h restore reserve exceeds the cap, the load does not start and is recorded as unrun (budget). A confirm block starts only if all of it fits.
- **Other stops:**
  - a token-ID mismatch in a champion load (T, W-B, X-D4b);
  - two failed starts of a required load;
  - the claim revoked or production needing the GPU;
  - any router or daily change by others.
- **Operations:**
  - **LEFT:**
    - Check the fallback first: analyst-station and vision-station must answer through agent-station on RIGHT.
    - Take a fleet claim by GPU UUID and stop only glmf-daily (`docker stop --time 180`).
    - Restore with `docker start`, prove both routes through LEFT, compare identities, then release.
  - **RIGHT:**
    - Check that agent-station falls back to analyst-station on LEFT.
    - Take a claim and stop only dsv41-daily on the GB300. The RTX PRO 6000 sidecar is not touched.
    - Restore with `docker start`, prove agent-station through RIGHT, then release.
- **Public outcome:** under `outcomes/selfopt-d-field`, with all round values, gates and dispositions. Private hosts, paths, device IDs and credentials stay out.
