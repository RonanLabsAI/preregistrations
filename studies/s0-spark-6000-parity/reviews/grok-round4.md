Point 4 stays open. Points 1, 2, 3, 5, and 6 are closed. One new should-fix: the hardware histogram gate treats idle `Running: 0` samples as overlap, so disjoint active batch ranges still pass.

**(A)**

**1. RESOLVED.** Declined, and the host evidence holds. `s0_job.py` records `docker inspect` `.Image`, and `check_treatment` compares it with `BASE_IMAGE` (`sha256:3af6a45b060d681aa6254200f87859d1bb3c1f962afb3d15540b4b31d4a41063`) on control units and with `BI_IMAGE` (`sha256:c92aec79…`) on every batch-invariant unit. The 2026-11-28 test reads that same field: a container created with the harness argv shows `.Image` equal to the pinned base digest, and `docker image inspect <base>@sha256:3af6… -f {{.Id}}` returns the same digest. On Docker 29 with the containerd image store, the local image ID of this pulled image is the manifest digest. `.Image` is fixed at create, so a created container is a valid probe. Any other image still fails the unit.

**2. RESOLVED.** Accepted in part, and the evidence holds. The same test shows `Path=vllm` and `Args` = `serve`, `--model`, path, `--served-model-name`, name, `--host`, `0.0.0.0`, `--port`, `8000`, then the spec's `serve_args`. `strip_model_and_port` drops one leading `serve` and the `--model` / `--host` / `--port` pairs, then requires exact equality with `--served-model-name`, the served name, `COMMON`, and `MEM[mem]`. That list is the card's flag order, including `0.85` and no KV-byte flag on the V1 arm. Exact order matches a frozen harness order, and an extra, missing, or reordered flag still fails the unit. The tested argv has no `-m` token.

**3. RESOLVED.** The flag was correctly left out. The card now says a PASS licenses pooling only of outputs from this protocol's fixed-KV configuration on these two platforms, and that the October runs are outside that license because their server flags were not logged. Reference comparisons stay out of the verdict; `pooling_allowed` is true only when the verdict is `PASS`. A flag that became true on 1,179/1,179 would license runs this card refuses. The Qwen sentence extends the earlier cross-device finding to the GB10 / RTX PRO 6000 pair; the license sentences and "a match or a mismatch there decides nothing" keep the October files out of the pool.

**4. PARTLY.** The 64-vs-32 pair is relabelled "hardware at unequal lower concurrencies" in the card and in `reported()`, and it is not a verdict input. For `hardware` (`m-spark-c128` vs `m-6000-c128`), `comp` sets `INCOMPLETE` when the bins-of-16 histograms share no key. That test does not catch disjoint active ranges, for the reason in (B).

**5. RESOLVED.** The parser is `GPU KV cache size: ([\d,]+) tokens` only. Commas are stripped, and a unit with any count other than one distinct value fails. `CPU KV cache size` does not match. Two `kv16g` units with different sizes make that comparison `INCOMPLETE`. The V1 arm is `gmu085`, so it is not required to equal the 16 GiB pool.

**6. RESOLVED.** `--timeout` defaults to `2700.0`, and the contract's lost-work bound is 45 minutes (2,700 s). The October Spark tail was 181 s for a 4,096-token Qwen3-1.7B reply; Granite 8B is about five times that parameter count, so that tail sits on the old 900 s default. Concurrent in-flight requests share one wall-clock window, so GPU time discarded at an interrupt is bounded by the same 2,700 s.

**(B)**

Should-fix. With the frozen flags, vLLM's default engine stats stay on and write `Running: 0 reqs` while the server is idle. `server_log_facts` bins every `Running: N reqs` sample as `min(N // 16 * 16, 1024)`, so both c128 logs contain bin `"0"`. `ha & hb` is then non-empty, and the hardware `INCOMPLETE` branch does not run. The two units can occupy disjoint positive bins and still be judged `IDENTICAL` or `DIFFERENT`. A mismatch in that case becomes `HARDWARE-DIFFERENT` even though the devices never ran a comparable batch. Ignore non-positive running counts, or require a shared bin with a running count of at least 1.

Points remain open: point 4.
