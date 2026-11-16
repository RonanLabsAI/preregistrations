# Study C verdict

**Two arms admitted. Champion candidate: D4b** (DFlash2 drafter, linears at 16 bits,
`--drafter-bits 0`). Changing the champion is the PI's decision.

Against the drafts-on champion (MTP drafting), D4b's paired ratio and two-sided t95 (n = 10 pairs per cell):

| Cell | Ratio | t95 |
|---|---|---|
| C1 | 1.116 | [1.115, 1.118] |
| C4 | 1.211 | [1.210, 1.212] |
| C16 | 1.153 | [1.150, 1.155] |

D4a (the same drafter, 4-bit linears) also admitted, at C16 1.147 [1.143, 1.150]. It ranks second
on the C16 lower bound.

All token IDs matched the temperature-0 reference in every load and arm. Both adjacent A loads
stayed inside their drafts-on bands, with CV at most 1.5%.

D2a (`--mtp-confidence 0.5`) advanced from the screen but its phase-2 block did not fit the
budget. It is **unrun (budget)**, not a null.

Protocol: [PROTOCOL.md](../../protocols/selfopt-c-serving/PROTOCOL.md), SHA-256
`0f088cd7a8ad7ae9b4116acda8d4e125978b3607525f3d94ca79add77f918691`, bound at merge
`191b7a48412ee60647c5c3b5645553b14cf14c3b`.
