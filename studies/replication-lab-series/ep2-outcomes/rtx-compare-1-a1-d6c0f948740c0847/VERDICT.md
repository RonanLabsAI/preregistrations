# Verdict: CIG-SFT versus matched SFT (replication episode two)

Preregistration: https://github.com/RonanLabsAI/preregistrations/blob/6e472349b92392a6e4cd7570784acee7afbdab59/studies/replication-lab-series/ep2-amendment-a2-2026-11-14.md

Classification under the preregistered rule: **supported**.
CIG minus SFT Avg@16 was 23.69 points (paired bootstrap 95% CI [21.29, 26.11], n=500 problems) against the paper's 26.9, from receipts/results.jsonl.
CIG minus SFT Pass@16 was 6.40 points (95% CI [3.00, 9.80], n=500 problems), from receipts/results.jsonl.
Arm base Avg@16 was 46.3 points (n=500 problems) against the paper's 39.0, from receipts/results.jsonl.
Arm sft Avg@16 was 42.3 points (n=500 problems) against the paper's 41.5, from receipts/results.jsonl.
Arm cig Avg@16 was 66.0 points (n=500 problems) against the paper's 68.4, from receipts/results.jsonl.

## What this does and does not establish

It tests the authors' released code and recipe at one seed on the frozen evaluation set with a fixed prompt and grader. It does not test the other five benchmarks, other seeds, the larger-model or code results, or a separately tuned SFT baseline.
