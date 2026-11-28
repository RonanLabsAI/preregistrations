# Review record for R1 protocol v1

Author: Claude (Anthropic), research-program session. Reviewer: Grok 4.7 via the Grok Build CLI (xAI), a different
model family. Gemini, the usual reviewer, had no quota on 2026-11-28.

Rounds, all on 2026-11-28 (PT):

| Round | Input | Review | Resolution |
|---|---|---|---|
| 1 | draft card r1 | `grok-round1.md` (4 blockers, majors) | card r2 (folded into the card before the stage-card freeze) |
| 2 | frozen card + harness | `grok-round2.md` (2 blockers, majors) | Deviation 3 in the protocol |
| 3 | card + Deviation 3 + harness | `grok-round3.md` (2 new blockers, 2 majors) | Deviation 4 in the protocol |
| 4 | card + Deviation 4 + harness | `grok-round4.md` | none needed: "no open blockers" |

Before round 1 the draft was also checked against the literature (Rane 2611.29663 read in full; 2611.15671,
2605.19537, 2611.06977, 2609.04748, 2510.22876 and DriftBench), a Cortex corpus consult and a web and vLLM-tracker
novelty search; the findings are summarized in the protocol's "Prior work" section.

Round 4 leaves one descriptive item "partly": the analysis entry point does not yet call the H2 interval rule, the
set-size curve or the rescore check. These are descriptive, not confirmatory; their code is added under a dated
amendment before those analyses are run.
