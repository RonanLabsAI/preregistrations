# Study C: serving settings on GB300, drafts-on baseline

Run 2026-11-15, 14:46-20:05 PT, on one GB300.

- **Engine:** TensorFold `7c1bc0be`, frozen champion binary (server SHA-256 `1711a2c9...9c92`), no engine changes.
- **Model:** GLM-5.3-Flash EXL3 4 bpw, revision `078455ff`.
- **Protocol:** [PROTOCOL.md](../../protocols/selfopt-c-serving/PROTOCOL.md), bound at merge `191b7a48`. It was published before any GPU work.

## Result

Swapping the built-in MTP drafter for the DFlash2 drafter (`incoai/GLM-5.3-Flash-DFlash2` at
`bf582e4e`) gives 12-21% more decode throughput. The output is bitwise identical.

Both drafter precisions admit. **D4b (16-bit linears) is the champion candidate**, by the larger
C16 lower bound.

| Arm | Cell | A mean (tok/s) | B mean (tok/s) | Paired ratio | t95 | Admit |
|---|---|---|---|---|---|---|
| D4b | C1 | 125.6 | 140.2 | 1.116 | [1.115, 1.118] | yes |
| D4b | C4 | 196.8 | 238.4 | 1.211 | [1.210, 1.212] | yes |
| D4b | C16 | 287.8 | 331.7 | 1.153 | [1.150, 1.155] | yes |
| D4a | C1 | 125.7 | 140.4 | 1.117 | [1.116, 1.118] | yes |
| D4a | C4 | 196.9 | 237.6 | 1.207 | [1.206, 1.207] | yes |
| D4a | C16 | 287.8 | 330.0 | 1.147 | [1.143, 1.150] | yes |

Phase 2 used A-B-B-A loads with one warm and five retained rounds per cell and load. That gives 10 pairs, and the
t95 interval has df 9. A is the drafts-on champion. The A loads were P2-A1, A2 and A3, all inside the frozen bands, all with CV
at most 1.5%.

The gain comes from acceptance. Here is the measured draft acceptance on retained rounds:

| Cell | A (MTP) acceptance | DFlash2 acceptance |
|---|---|---|
| C1 | 0.55 | 0.63 |
| C4 | 0.50 | 0.68 |
| C16 | 0.67 | 0.83 |

Cost is device memory: the 16-bit drafter takes 2.11 GiB, the 4-bit one 0.93 GiB. The 4-bit arm
is within 0.6% of the 16-bit arm at every cell, so the PI may prefer D4a if memory matters
more than the last half percent.

## Phase 0: baseline and reference

There was one load. Each cell had a warm round plus 5 retained rounds, alternating A (drafts on) and R
(drafts off). Every ID matched.

| Cell | A mean | A CV | R mean | R in original band | A/R paired (t95) |
|---|---|---|---|---|---|
| C1 | 125.33 | 0.29% | 81.78 | yes [80.12, 83.39] | 1.533 [1.527, 1.538] |
| C4 | 197.11 | 0.02% | 171.96 | no band | 1.146 [1.146, 1.147] |
| C16 | 287.57 | 0.19% | 273.14 | yes [267.33, 278.24] | 1.053 [1.050, 1.055] |

The frozen drafts-on bands were A ±2%.

## Phase 1 screen

Each arm got one load, a warm round plus 3 retained rounds per cell, against phase-0 A. Every arm's IDs passed.

| Arm | Setting | C1 | C4 | C16 | Advance |
|---|---|---|---|---|---|
| K- | `TF_GLM_SHARED=0` (sensitivity control) | | 0.620 | 0.427 | control passes (< 0.95) |
| D2a | `--mtp-confidence 0.5` | 1.026 | 1.064 | 0.941 | yes (rank 3) |
| D2b | `--mtp-confidence 0.8` | 0.993 | 1.058 | 0.938 | no (4th; at most 3 advance) |
| D3 | `TF_GLM_COPIES=1` | 1.004 | 1.027 | 1.003 | no |
| D4a | DFlash2, `--drafter-bits 4` | 1.120 | 1.207 | 1.148 | yes (rank 2) |
| D4b | DFlash2, `--drafter-bits 0` | 1.119 | 1.210 | 1.153 | yes (rank 1) |
| S1a | `--parallel 1` | 1.001 | | | no |
| S1b | `--parallel 4` | | 0.999 | | no |
| S7 | `TF_GLM_L2PF=1` (123 rows at startup) | 0.996 | 0.997 | 1.000 | no |
| S8 | `TF_KV_PAGES=on` | 1.000 | 0.999 | 0.999 | no |

S2-S6 are unrun by prospective source reading, as the protocol states. They act only on prompts
over 512 rows, and every corpus prompt is 85-97 tokens.

D2a is **unrun (budget)** in phase 2. Its whole block did not fit in the remaining budget. The screen suggests MTP confidence 0.5 helps at C4 and hurts at C16, but that is
screen data only, not evidence.

These results bound only this corpus: short prompts, no prompt cache, and C1/C4/C16. Long-prompt
prefill settings were not tested.

## Operations

- **GPU use:** 5.31 of the 6.0 GPU-hours. SM clock was steady at 2,070 MHz in every busy sample, 55 C maximum, and no throttle reasons were set.
- **Restore:**
  - The daily was healthy in 166.6 s.
  - Both routed fallbacks answered on the first probe.
  - Daily, second-Station and production identities matched the pre-study snapshots.
- **Procedure notes:**
  - The client posted directly to the engine instead of through the research router.
  - The engine listened on loopback only.
  - Phase-2 blocks were sequenced by `receipts/run_p2.py`, which applies the protocol's whole-block budget rule.

## Receipts

All under `receipts/`:

- `cells/`: every round, with per-request token hashes and speculative counts. Raw token streams stay private.
- `analysis/`: phase-0, screen and confirm outputs from the bound `analyze.py`.
- `loads.json`: per-load dispositions and load times.
- `plans/`: plans as run.
- `operations.json`: timings, restore, telemetry summary.

Private hosts, paths, device IDs and credentials are omitted.
