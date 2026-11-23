# LingSante-3.1-flash as a local medical fact-checker on two DGX Station GB300s

Frozen 2026-11-22 PT, before any LingSante request. Times are PT (America/Los_Angeles).
Stephen (PI) directed the study on 2026-11-22: "lets try it on the fused stations ... add this testing to your
next station window". Manual PI-directed execution under fleet claim `blackhawkhair-r3-fused-20261122`
(preemptible). Every outcome is recorded here, including nulls, failed boots and unrun dispositions.

## Question

Can an open-weight medical model running locally on two desktop GB300 Stations replace a frontier model
(Claude Opus 5.5) as the check pass that flags factual, regulatory and safety errors in draft
patient-education pages?

## Source claim

inclusionAI, 2026-11-22: LingSante-3.1-flash (560B total / 25B active, MIT) "leads on HealthBench
Professional (65.72) ... outperforming GPT-6 Astra, Claude Opus 5.5 and Gemini 4 Argon". We test a narrower,
practical claim: flagging errors in cited medical drafts.

## Arms

- **C (reference): Claude Opus 5.5** check pass, 6 subagents x 5 pages, run first, blind to L.
- **L: LingSante-3.1-flash-MXFP4** (`inclusionAI/LingSante-3.1-flash-MXFP4`, 174 files, 313,232,398,612 bytes;
  config.json and model.safetensors.index.json sha256 in binding.json), SGLang image
  `lmsysorg/sglang:dev-ling-3.1-flash` (sglang 0.0.0.dev1+g4728eda79, humming-kernels 0.1.12), TP2/EP2 across
  two DGX Station GB300 (one GB300 each, one 400G rail), MXFP4 experts with FP8 activations, NEXTN off,
  thinking on, temperature 1.0 / top_p 0.95 / top_k 20 (model card), one request per page, max 32 concurrent.
- Same prompt for both: `CHECK_PROMPT.md` (sha256 in binding.json). Same inputs: 30 private draft pages
  (exported packets; per-file sha256 in binding.json; content not public: unpublished clinical drafts).

## Metrics (frozen)

1. Flags per page; same-sentence overlap of L with C.
2. **Recall of C's core flags**: share of C's `misread-result` + `regulatory` flags that L flags on the same
   sentence id, with Wilson 95% CI, denominator = C core flags.
3. **Adjudicated precision**: random sample (seed 20261122) of up to 20 L-only and 20 C-only flags, judged
   blind (source hidden) by a fresh Claude Opus 5.5 judge given the page packet; precision per arm with Wilson
   95% CI. Judge-model overlap with arm C is a stated bias.
4. JSON validity rate; wall time per page; decode tok/s per request (reported, not gated).

## Decision rule (frozen)

L can replace C for the full site only if adjudicated precision(L) >= precision(C) - 0.10 **and** core recall
>= 0.70. Otherwise C stays.

## Budget, kill and interruption

- Boot budget 45 min; two failed starts = **unrun (boot)**, recorded with the error.
- Run budget 60 min for 30 pages; a page with no valid JSON after 2 attempts counts as a miss for L.
- Production (another fleet claim) takes priority: an interruption is recorded and the run is not resumed
  under a changed setup.
