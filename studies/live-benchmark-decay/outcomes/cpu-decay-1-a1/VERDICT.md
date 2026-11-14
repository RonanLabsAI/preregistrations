# Verdict: live-benchmark decay (LiveCodeBench primary, LiveBench secondary)

Preregistration: https://github.com/RonanLabsAI/preregistrations/blob/e21101fa15381704d40972fdffa9d7ab4f01b823/studies/live-benchmark-decay/amendment-a1-2026-11-14.md

## Primary (LiveCodeBench, adjacent calendar quarters, difficulty-adjusted)

Classification under the preregistered three-point tolerance: **not comparable**.
Mean absolute difficulty-adjusted drift between adjacent quarters was 6.04 points (95% CI [5.25, 8.78], n=196 model-pair cells), from receipts/results.jsonl.
Unadjusted mean absolute drift was 6.21 points (95% CI [5.72, 9.61], n=196 model-pair cells), from receipts/results.jsonl.
Median Kendall tau-b between adjacent-quarter model rankings was 0.850 (95% CI [0.74, 0.83], n=7 quarter pairs), from receipts/results.jsonl.
The panel covered n=28 models and n=1043 problems in the included quarters, from receipts/results.jsonl.

## Secondary (LiveBench, adjacent releases, task-matched)

- coding 2024-06-24 00:00:00 to 2024-07-26 00:00:00: adjusted mean absolute drift 4.62 points (95% CI [5.71, 13.68], n=154 models), classified not comparable, from receipts/results.jsonl.
- instruction_following 2024-06-24 00:00:00 to 2024-11-25 00:00:00: not analyzed (fewer than 5 complete models or 10 task-matched questions; n=163 complete models).
- language 2024-06-24 00:00:00 to 2024-11-25 00:00:00: adjusted mean absolute drift 8.96 points (95% CI [5.93, 17.79], n=38 models), classified not comparable, from receipts/results.jsonl.

## Exploratory (not a preregistered test)

Post-minus-pre release-marker gaps use LiveCodeBench's own leaderboard markers, not verified training cutoffs.
- Claude-3.5-Sonnet-20241022 (marker 2024-03-31): -16.39 points (95% CI [-20.78, -11.74], n=626 post-marker problems), from receipts/results.jsonl.
- DeepSeek-R1-0528 (marker 2024-06-30): -14.30 points (95% CI [-17.63, -11.01], n=505 post-marker problems), from receipts/results.jsonl.
- DeepSeek-V3 (marker 2024-06-30): -29.39 points (95% CI [-34.08, -24.63], n=505 post-marker problems), from receipts/results.jsonl.
- EXAONE-4.0-32B (marker 2024-04-01): -10.58 points (95% CI [-14.29, -6.91], n=616 post-marker problems), from receipts/results.jsonl.
- Grok-3-Mini (High) (marker 2024-03-01): -7.41 points (95% CI [-12.41, -2.69], n=661 post-marker problems), from receipts/results.jsonl.
- OpenCodeReasoning-Nemotron-1.1-32B (marker 2024-04-01): -10.53 points (95% CI [-14.02, -6.51], n=616 post-marker problems), from receipts/results.jsonl.
- OpenReasoning-Nemotron-32B (marker 2024-04-01): -10.49 points (95% CI [-14.02, -6.86], n=616 post-marker problems), from receipts/results.jsonl.
- Qwen3-235B-A22B (marker 2024-06-30): -17.26 points (95% CI [-21.50, -13.15], n=505 post-marker problems), from receipts/results.jsonl.
- XBai-o4-medium (marker 2024-04-01): -15.81 points (95% CI [-20.05, -11.70], n=616 post-marker problems), from receipts/results.jsonl.

## What this does and does not establish

It measures score comparability across problem windows for a fixed set of models on two public live benchmarks. It does not identify why windows differ beyond the difficulty and task mix it adjusts for. It does not establish contamination. The snapshots are frozen public files; the live leaderboards may have changed since.
