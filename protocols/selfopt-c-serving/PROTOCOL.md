# TensorFold study C: serving settings on the GB300 champion, drafts-on baseline

Frozen 2026-11-15 PT, before any study GPU request. Times are PT (America/Los_Angeles).
Stephen (PI) approved on 2026-11-15 at 14:32 PT: "Freeze the prereg with drafts-on as baseline
and run it". This revises the unpublished draft C (tf-prep commit c688ba2): the baseline is now
drafts-on, drafts-off is a recorded reference cell, and the phase budget is re-derived. Everything
else in the draft is kept unless this file says otherwise. No engine source changes: every arm is an
existing flag, request field or environment switch of the frozen champion binary. Manual PI-approved
execution under a revocable, preemptible device claim; no scheduler guarantee is inferred. Every
outcome is recorded, including nulls, gate stops and unrun dispositions.

## Question

Does an existing serving setting earn at least **1.05x** throughput over the **drafts-on** champion
at C1, C4 or C16, with every reply's token IDs bitwise equal to the drafts-off token reference at
temperature 0?

## Frozen champion and inputs (unchanged from ../selfopt-g3-1; hashes in binding.json)

- Engine `ronanlabs/sm103-enable` @ `7c1bc0be3df413f43c65fb278699b629a44e7a90`, served binary
  sha256 `1711a2c9...7c9c92`. Native FP off. One rank (`TF_GLM_RANK=0 TF_GLM_WORLD=1`), GB300 sm_103.
- Model `Mia-AiLab/GLM-5.3-Flash-EXL3-4bpw-TensorFold` @ `078455ffe647` (`glm5_next`, 1 MTP layer).
- Recipe `serve <model> --backend cuda --context 40960 --parallel 16 --prompt-cache-gib 0
  --no-thinking` (recipe hash `f8d2add5...e84`). MTP and suffix copies load by default.
- Corpus: the 16 public prompts of `reference-corpus.json` (85-97 prompt tokens each), 512 output
  tokens, temperature 0, seed 1234, thinking off. Token reference: `TOKEN-REFERENCE.json`
  `long_plain_decode` (drafts-off IDs, one list per prompt, concurrency-independent).

## Baseline and reference cell

- **A (baseline) = the champion recipe with every request sending `"draft": true`** and no other
  draft setting, so the champion's default drafting applies (MTP head plus suffix copies, lane
  depth, no `--mtp-confidence`, no drafter). Every arm is measured against A in the same request
  mode (drafted).
- **R (reference cell) = the same load with `"draft": false`.** It is recorded at C1, C4 and C16 in
  phase 0 only. R is not a baseline and no arm is scored against it. The drafted/plain ratio A/R is
  reported as a descriptive measurement (the old arm D1), not as a candidate.

## Settings that exist (checked at `7c1bc0be`; file:line from the draft survey)

As in draft C: request `draft` (`zig/src/server/chat.zig:182-183`), `--mtp-confidence`
(`cli.zig:114`), `--drafter`/`--drafter-bits` (`cli.zig:107-108`; refused with `--mtp-confidence`,
`native/cuda_refusal.zig:16`), `TF_GLM_COPIES` (`cuda/priority_copies.zig`), `--parallel`,
`--decode-share`, `TENSORFOLD_PREFILL_*`, `TF_GLM_PROMPT_STEPS`, `TF_GLM_SHARED`,
`TF_GLM_CHUNK_SOLO`, `TF_GLM_L2PF`, `TF_GLM_PDL`, `TF_KV_PAGES`. `--mtp-drafts` is refused for GLM,
so no direct draft-length arm exists. Exclusions are unchanged: `--kv-cache fp8`,
`--prompt-precision int8` and `TF_GLM_REQUEST_STARTS` change arithmetic; `--prompt-cache-gib` would
give a cache-hit artifact on a repeating corpus; `--lane-kernels` is not native; overlap and RoCE
switches are TP-only; `--context` changes admission, not decode.

## Arms (frozen before any timing; one setting each; no tuning after timing)

All arms run in drafted mode against A.

| Arm | Change from A | Cells |
|---|---|---|
| D2a | `--mtp-confidence 0.5` | C1, C4, C16 |
| D2b | `--mtp-confidence 0.8` | C1, C4, C16 |
| D3 | `TF_GLM_COPIES=1` (root/context copies) | C1, C4, C16 |
| D4a | `--drafter <DFlash2> --drafter-bits 4` | C1, C4, C16; only if that load opens and answers |
| D4b | `--drafter <DFlash2> --drafter-bits 0` | C1, C4, C16; only if that load opens and answers |
| S1a | `--parallel 1` | C1 |
| S1b | `--parallel 4` | C4 |
| S7 | `TF_GLM_L2PF=1` (default knobs) | C1, C4, C16; only if the startup log prints a `TF_GLM_L2PF: <n> rows` line with n > 0, else unrun |
| S8 | `TF_KV_PAGES=on` | C1, C4, C16 |
| K- | `TF_GLM_SHARED=0` | C4, C16. **Sensitivity control only, never a candidate**: its C16 screen ratio must be below 0.95, or the harness is declared insensitive and the study halts |

DFlash2 = `incoai/GLM-5.3-Flash-DFlash2` @ `bf582e4eacc1810f76656d1811693ff6c6737d2a`, local
`model.safetensors` sha256 `b038e1d9...9dac` (equal to the HF LFS sha256) and `config.json` sha256 in
binding.json. A D4 load that fails to open, or refuses, is recorded as unrun (two failed starts).

**Unrun by prospective source reading (not measured nulls):** the draft's S2 (`--decode-share 0.5`),
S3 (`TENSORFOLD_PREFILL_POLICY=fcfs`), S4 (`TENSORFOLD_PREFILL_ORDER=srpt`), S5
(`TF_GLM_PROMPT_STEPS=0`) and S6 (`TF_GLM_CHUNK_SOLO=4096`). At `7c1bc0be` the prefill share, budget,
policy and SRPT order act only on long prompts, those whose pass exceeds `Share.short_rows = 512`
rows (`zig/src/core/lane_host_fill.zig:73,150-200,266-298`), and the `--decode-share` chunk cap is
512 rows (`lane_host_fill.zig:146-147,302-306`). Prompt steps and solo chunks split prompts of 2048
rows or more (`glm/cuda_native.zig:368-372`, `glm/cuda_ranks.zig:20,94-99`). Every corpus prompt
is 85-97 tokens and fills in one pass (W26 receipts: one prefill width per reply), so these arms
cannot act on this workload. They belong to a long-prompt study. `TF_GLM_PDL=0` stays listed but
unrun (prior: slower); it is the fallback negative control only if K- cannot start.

## Workload, client and metrics (G3 conventions)

- The client runs on the Station and posts directly to the owned engine's loopback HTTP port
  (`/v1/chat/completions`). No router or route is created or changed. This differs from G3.1, which
  went through a research router; R's C1/C16 values are compared to the original bands only as a
  recorded drift check (below).
- Request body: the prompt as one user message, `temperature 0`, `seed 1234`, `max_tokens 512`,
  `draft` true (A, arms) or false (R), `return_token_ids true`,
  `chat_template_kwargs.enable_thinking false`. Each reply must report 512 completion tokens, 0
  cached prompt tokens, thinking off, and `tensorfold.drafts` equal to the requested mode.
- C1: the 16 prompts in corpus order, one at a time. Metric: per-reply native decode 511 /
  `tensorfold.seconds`, meaned over the 16 replies of a round.
- C4: four concurrent clients; client k (0-3) sends prompts 4k..4k+3 in order. C16: all 16 at once.
  Metric for both: 8192 completion tokens / client round wall seconds.
- Each cell on each load starts with one excluded warm round per request mode.
- Drafted replies' speculative counters (drafted, accepted, rounds), when the server returns them,
  are recorded with every round. GPU clocks, temperature, power and clock-event reasons are sampled
  every 2 s without a CUDA context.

## Numerical gate

- Every reply of every round, warm rounds included, in every arm, cell and mode must have token
  IDs **bitwise equal** to `TOKEN-REFERENCE.json` `long_plain_decode` for its prompt. The
  server-reported `token_sha256` must equal the SHA-256 of the comma-joined decimal IDs
  (`zig/src/server/tokens.zig:105-113`).
- Drafting must be lossless. Any mismatch fails that arm (all its cells), is recorded with the first
  differing position, and gets no timing interpretation. The load is stopped at the first mismatch.
  Nothing is widened.

## Phase 0 (one load of the champion recipe)

At C1, then C4, then C16: one warm A round and one warm R round, then 5 retained rounds of each,
alternating A, R, A, R, ... within the same load.

- **Baseline gates (A).** All IDs equal the reference, and the sample CV of the 5 retained round
  values is at most 1.5% at each cell. **If A fails either gate at any cell, the study stops**: the
  failure is reported and nothing is tuned around it.
- **New frozen drafts-on control bands:** A's 5-round mean ±2% at C1, C4 and C16, frozen when
  phase 0 ends and never widened. A later A load whose 5-round mean falls outside its band halts
  scoring at that cell.
- **R (reference) checks.** R's IDs must equal the reference; a mismatch stops the study, since the
  reference or client is then broken. R's C1 and C16 means are compared to the original drafts-off
  bands [80.123, 83.394] / [267.328, 278.239]. Being outside a band is recorded and flagged for the
  PI, but does not halt, because R is a reference cell and the client path changed. R's C4 mean and
  CV are recorded. R's CV over 1.5% is recorded.
- Descriptive A/R paired ratio: the paired log-ratio mean over the 5 alternating pairs, Student
  t95 df4 (critical 2.776445105), both endpoints back-transformed.

## Phase 1 screen (one load per arm, drafted only)

- One warm round plus 3 retained rounds per applicable cell. Screen ratio = the arm's 3-round mean
  divided by phase-0 A's 5-round mean at the same cell. This is a screen, not evidence.
- K- runs first. If its C16 ratio is not below 0.95, the study halts as insensitive.
- An arm advances if its screen ratio is at least 1.03 at any applicable cell and its IDs all pass.
  At most **3** arms advance, ranked by their largest screen ratio; exact ties at 3 decimals go to
  the arm listed earlier in the table. S1a and S1b are separate arms.

## Phase 2 confirm (each advanced arm, in rank order)

- Loads in A-B-B-A order, 1 warm plus 5 retained rounds per applicable cell and load, drafted only.
  Consecutive arms share the A load between them (A-B1-B1-A-B2-B2-A ...). An A load measures the
  union of the cells of the arms next to it.
- Pairs: round i of the first A load with round i of the first B load, and round i of the second B
  load with round i of the following A load: 10 pairs per cell.
- Statistic: mean of the paired log ratios (B/A), two-sided Student t95, df 9 (critical
  2.262157), both endpoints back-transformed.
- **Admission:** the t95 lower bound is above **1.05** at a cell where the arm applies, every ID
  passes in phase 1 and phase 2, both adjacent A loads stay inside the drafts-on bands at that cell,
  and A's CV stays at most 1.5%. If several arms admit, the champion candidate is the one with the
  largest lower bound at C16, then C1. Combinations are not tested in C. Admission makes a champion
  candidate; changing the champion is the PI's decision.

## Budget and stops

- **GPU cap: 6.0 GPU-hours** of exclusive LEFT GB300 reservation, from claim to release, including
  loads, waits, CPU checks and restore. The study stops at the cap whatever its state.
- Re-derived estimate for the drafts-on baseline. Load to health is about 1 min (W26: loaded in
  53 s); 2 min per load is budgeted with stop and checks. Round estimates: plain C1 102 s, C4 51 s,
  C16 30 s (W26; C4 from W14b's 2.0x C4/C1 ratio); drafted C1 74 s, C4 40 s, C16 35 s (W14b
  dev-screen ratios 1.74x at C1 and 1.26x at C4; C16 drafted is unmeasured, so plain + 17% is used).
  - Phase 0: 6 A and 6 R rounds at three cells, plus one load: about 35 min. With the preflight,
    claim, fallback checks and daily stop (8 min): **0.72 h**. The draft's 0.6 h assumed plain A.
  - Phase 1: seven three-cell loads at 11.9 min, S1a 6.9 min, S1b 4.7 min, K- 7.0 min:
    **1.70 h**. Dropping S2-S6 saves about 0.6 h against the draft's 2.6 h.
  - Phase 2: a three-cell load is 16.9 min. The first arm costs 4 loads (1.13 h) and each
    further arm 3 (0.85 h). Three arms: **2.82 h**. The draft's 2.4 h assumed plain round times.
  - Restore, route checks and postflight: about 0.1 h. Total about **5.34 h**, leaving 0.66 h.
- Budget rule: before any load, if elapsed + that load's estimated cost (from measured round times
  once available) + a 0.25 h restore reserve exceeds 6.0 h, no new load starts and the remaining
  arms are recorded as unrun (budget). A phase-2 block starts only if its whole A-B-B-A cost fits
  under the same rule.
- Other stops: A fails its ID or CV gate in phase 0; an ID mismatch in any A load or in R; an A
  band failure (halts scoring at that cell); a K- sensitivity failure; two failed starts of one arm
  (recorded as unrun); a second consecutive failure of the same step; the claim revoked or
  production needing the GPU; any production router or daily change by others. A step that needs
  sudo and gets "password required" stops the study.

## Operations and interruption contract

- Before the claim: check the analyst-station and vision-station fallbacks through RIGHT with the
  RIGHT daily up and LEFT still serving. Then take a preemptible fleet claim for the LEFT GB300 by
  GPU UUID, plus a shared, lifecycle-free claim covering the RIGHT devices used by routed probes.
  Snapshot the LEFT daily, RIGHT containers and production router identity.
- Stop only the LEFT daily, with its own tooling (`docker stop --time 180` on the preserved
  container), then verify both LEFT-down fallbacks answer through RIGHT. Never touch RIGHT GPUs,
  RTX PRO 6000s, the production router or any route. The engine process is pinned to the LEFT GB300
  by UUID. A concurrent idle-priority file copy on the Station is left running; its effect on loads
  is recorded (loads are excluded from all metrics).
- Restore: stop the owned engine, `docker start` the preserved daily, time to health and to both
  routed answers through LEFT, compare the daily, RIGHT and production identities with the
  snapshots, then release the claims. Restore runs on any stop, error or the cap.
- Public outcome: an outcome package under `outcomes/<this protocol>` with the protocol SHA-256 and
  the immutable commit URL, all round values, gates, dispositions, GPU-hours and restore times.
  Private host, path, network, device-UUID and credential details stay out of public files.
