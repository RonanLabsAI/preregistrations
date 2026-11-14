# Vendor headline check v1: amendment A1 (2026-11-14 PT)

Amends protocol v1 (`protocol-v1-2026-11-14.md`, commit `58f2aa354cf6d252b5f94b143afaf4690945d4bc`). Everything not listed here is unchanged and remains binding. No timing measurement has been taken.

## Changes

1. Verdict wording only: numbers appear only in sentences that carry their denominator and interval, and the environment fingerprint is reported in the package manifest instead of the verdict text, as the result-package validator requires. Found by validating a package produced against a local fake router; no request reached the model.
2. Frozen environment fingerprint becomes `gb10|drv580.142|ollama0.32.4|gpt-oss120b-blob@6be6d66a3f54|ctx65536|np8|fa1|litellm1.104.1|route=gpt-oss|c1|code@81f6beab3c21`.

Execution follows public commit of this amendment.
