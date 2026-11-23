# Verdict: LingSante-3.1-flash does not replace the frontier check pass

Protocol: ../../protocols/lingsante-checker-gb300/PROTOCOL.md (frozen 2026-11-22 22:11 PT; Amendment 1 at 23:52 PT,
before scoring). Run 2026-11-22 23:16 PT to 2026-11-23 02:50 PT on two DGX Station GB300s, TP2/EP2, MXFP4 experts.
Status: preregistered locally; timestamps not externally verifiable (local commits b457c32, 3edd7d9).

**Decision rule not met; the Claude check pass stays.**

- Primary arm L (frozen settings, thinking on, 32,000-token reply cap): answered 8 of 30 pages, n=30. The other 22
  spent the whole cap on reasoning (prompts 11K-37K tokens). Median 1,245 s and 26,340 completion tokens per answered
  page, about 33 tok/s per request at 8 concurrent.
- On those 8 pages L found 7/21 of the Claude checker's misread-result and regulatory flags (33%, Wilson 95% CI
  17%-55%).
- Secondary arm L-nothink (Amendment 1, thinking off, misses only): 22/22 answered, median 308 s. L + L-nothink over
  all 30 pages: 670 flags against Claude's 229; core recall 57/109 = 52% (95% CI 43%-61%), below the 70% bar.
- Blind adjudication of disagreements (seed 20261122, n=20 per arm, source hidden, Claude Opus 5.5 judge): Claude-only
  flags 20/20 valid (100%, 95% CI 84%-100%); LingSante-only flags 4/20 valid (20%, 95% CI 8%-42%). The judge shares a
  model family with arm C, a bias stated in the protocol; it cannot account for a 20/20 vs 4/20 split.

What it teaches: on long, citation-dense medical documents this model's reasoning does not terminate inside a 32K
reply budget most of the time, and with reasoning off it over-flags (about 3 flags for every Claude flag, most
invalid). Benchmarks on short medical questions did not predict this. Inputs are unpublished clinical drafts and are
not released; per-file hashes are in the protocol's binding.json.
