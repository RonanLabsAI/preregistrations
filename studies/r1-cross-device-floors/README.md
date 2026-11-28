# R1: do greedy-evaluation noise floors transfer across devices?

Status: **frozen protocol, not yet executed** (counted runs start only after this record is merged).
Protocol: `protocol-v1-2026-11-28.md`, SHA-256 `b0ea44aac712308b30beda7047a905605ef2738aad63de1cad142b564c6c597e`. It is byte-identical to the lab's canonical card and
includes Deviations 1-4, all logged before any counted R1 output. Approved by the PI 2026-11-28 ("approve as is").

## Question and source claim
Rane (arXiv 2611.29663, 22 Nov 2026) measured greedy-evaluation noise floors (restart, reorder) and batch-invariant
bitwise identity on one consumer GPU and names one GPU as a limit; Blagojevich & Sunderland (2611.15671) likewise on one
GB10. R1 asks whether a same-config noise floor measured on one device holds on another device, and whether vLLM
batch-invariant outputs are byte-identical across four named NVIDIA Blackwell devices (GB300, RTX PRO 6000 Blackwell
Max-Q, GB10 DGX Spark, RTX 5080), including cold and warm restarts. Confirmatory test H1a: GB300 vs RTX PRO 6000,
Qwen3-4B-Instruct-2507, batch-invariant off, scheduler-matched offline generation, three warm restarts each: the
cross-device graded flip rate minus the mean of the two warm-warm floors, 95% cluster-bootstrap CI excluding 0. H1b:
pass/fail byte identity under batch-invariant mode for every within-device restart pair and every device pair. All
other arms are descriptive. Full design, statistics, predictions and stop rules: the protocol.

## Public inputs and identities
- Models: `Qwen/Qwen3-4B-Instruct-2507` rev `cdbee75f17c01a7cc42f958dc650907174af0554` (safetensors sha256
  75311d91..., 0b48adbb..., 7dd39ccc...); `Qwen/Qwen3-1.7B` rev `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`.
- Serving image: `vllm/vllm-openai:nightly-3f1bcfd82bb2e93fcd364fc7a5491718453a77e2`, digest
  `sha256:cb9bc64f0f038c2c2ebf13b31248fb87aca40c356c3d20f363be087cfcff1fcd` (arm64; vLLM 0.32.0rc1.dev190+g3f1bcfd82).
  The RTX 5080 (x86, WSL2) runs the same tag's x86 image if the tag publishes one (checked before its first run and
  stated in the outcome record); its digest is recorded per run and it is reported as a device-plus-build contrast
  (protocol Deviation 4).
- Items: `harness/configs/items/transfer_core.ids.txt` (2,244 items: GSM8K 659, MATH-500 250, IFEval 270, LiveCodeBench
  65, and a 1,000-item MMLU-Pro subsample stratified by category, rng 20261128) and `full_dev.ids.txt` (7,260 items,
  paper 2's set). The full item files are rebuilt from the public benchmarks by `harness/scripts/items.py build`; their
  SHA-256 values are in `harness/configs/items/SHA256SUMS` (item rows are not committed here: LiveCodeBench test cases
  make the files ~130 MB).
- Seeds: generation seed 0, temperature 0; order permutations 20261129, 20261130; bootstrap numpy default_rng(0),
  2,000 resamples; MMLU-Pro subsample rng 20261128.
- Plan: `harness/configs/plan.json` (124 runs incl. 10 uncounted warmups) and `runs.json`.
- Freeze manifests: `harness/freeze/R1_FREEZE*.sha256` (stage card 11:02 PT, stage code 11:37 PT, overlays fix1-fix4).

## Metrics, denominators, intervals
Per item and run pair: graded-outcome flip (primary), extracted-answer and text flips. Floors per kind (warm-warm,
cold-warm, cold-cold, same-process, order), equal-benchmark-weight means over the 2,244 items. Intervals: cluster
bootstrap within benchmark (MMLU-Pro category clusters; other items individually), conditional on the runs. Zero-flip
rates: cluster-bootstrap upper bound and Clopper-Pearson. Direction: exact McNemar; churn beside net accuracy.

## Budget and hardware
55 GPU-h cap (PI, 2026-11-28; Deviation 2). Devices: one DGX Station GB300, one RTX PRO 6000 Blackwell Max-Q (both
Grace hosts), one DGX Spark GB10, one desktop RTX 5080 (WSL2). Driver, CUDA, cuDNN, torch, vLLM, image digest, attention
backend (FLASH_ATTN forced), CUDA-graph mode and the engine configuration read back after start are recorded per run.

## Kill criteria and stop rules
A run aborts if any pinned engine setting reads back differently (block size 16, 2,048 KV blocks, bf16 KV, max 32
sequences, 8,192 batched tokens, context 8,192, prefix caching off, chunked prefill on, FLASH_ATTN, declared CUDA-graph
mode), if TF32 reads back on, if prefix-cache hits are non-zero, if the engine log is empty, if a tuned-config folder is
in effect, or if a run declared warm starts with an empty cache. The analysis aborts on unscored confirmatory items, an
item set other than the 2,244 transfer-core ids, unequal prompt-token totals in H1a, or a confirmatory run on another
image digest. The budget is re-projected after each device's first runs; over the cap, the protocol's drop order applies.

## Interruption contract
Runs are independent and resumable: the per-device queue (`harness/scripts/r1_queue.py`) skips a run whose
`run_meta.json` is clean, moves an aborted run aside and reruns it once, and stops on a second abort. Each device runs
under a preemptible fleet claim; production work takes priority, and a preempted run is rerun from scratch (never
merged with a partial output). An interrupted or unrun arm is recorded as such in the outcome record, not as a null.

## Smoke and runnable command (before admission)
On 2026-11-28, on the RTX PRO 6000 and GB300 with the pinned image: a 100-item timing smoke (RTX PRO 6000 92 s, GB300
22 s); two restarts per mode byte-identical within the device on those items (cross-device outputs were not compared
before this record); a live run of the final launch path (every pinned setting read back as requested); and the RTX PRO
6000's three uncounted warmups through the real queue (`r1_queue.py --device rtx6000 --arm warmup`, clean, 117-225 s
each). The 100 smoke items stay in the item set and are not cited for H1b (`harness/configs/smoke_items.txt`).
Runnable command per device: `R1_HOST_<DEV>=... R1_GPU_UUID_<DEV>=... python3 harness/scripts/r1_queue.py --device <dev>`
(hosts and GPU ids are supplied at run time and are not part of this record).

## Reviews
Four adversarial review rounds; see `reviews/README.md`.
