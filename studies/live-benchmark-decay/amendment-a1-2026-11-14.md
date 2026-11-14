# Live-benchmark decay: amendment A1 (2026-11-14 PT)

Amends protocol v1 (`protocol-v1-2026-11-14.md`, commit `58f2aa354cf6d252b5f94b143afaf4690945d4bc`). Everything not listed here is unchanged and remains binding.

## Why

A smoke run of the v1 harness (200 bootstrap resamples) after v1 was public exposed three execution defects: the command passes `--arm {arm}` but the v1 harness had no such flag; the result-package validator refuses binary (parquet) assets, so a frozen input could not travel with the package; and one verdict sentence carried a bare number. The smoke's classification output was visible to the operator. This amendment changes no analysis choice: windows, strata, statistic, tolerance, bootstrap, seed, models and items are as in v1.

## Changes

1. The harness accepts `--arm` (default `decay`) and its default seed is the unit seed (1), matching v1's stated arm, seed and unit.
2. The LiveBench judgments enter the package as `livebench_model_judgment.csv`, a deterministic export (question, model, task, category, turn, score; sorted; scores as Python float repr) of the pinned parquet, whose SHA-256 is unchanged (`35ad896970151776145c96b31c5ddb3a2749ea9a1d91e6b7f1a4c4c04735182a`, in `harness/inputs.lock.json`). The CSV's SHA-256 is `f85a5a56d92d6b67e611eaa39cf0e80eff46fbccd45c48091ec9a96f8155a66c`. The analysis reads the CSV.
3. Verdict wording: the tolerance is written in words so every numeric sentence carries a denominator or interval.
4. Frozen environment fingerprint becomes `cpu-x86_64|py3.12.15|numpy2.3.4|pyarrow21.0.0|lcb@f186ab50|livebench@9704e5da|code@d70e41ae9ba8`.

The pre-amendment smoke is recorded as a smoke attempt, not a result. Execution of the preregistered unit follows public commit of this amendment.
