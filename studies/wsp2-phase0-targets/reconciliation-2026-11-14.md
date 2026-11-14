# WSP v2 Phase 0: reconciliation record (2026-11-14 PT)

Product: `wsp2-phase0-targets`. Units `s1w-1`, `s2w-1`, `s3w-1`.
Disposition: **complete (historical, not prospectively preregistered)** for the targets, the stuck control and the held-out sets; **partial** for the trust-boundary harness. No unit is rerun.

Preregistered locally; timestamps not externally verifiable. The work described here was built on 2026-11-02 PT under a locally dated protocol and amendments, before this repository existed. This record does not make that work prospective; it records what exists, by hash, and what does not.

## What exists

- **Twelve PPO target policies** (four games, three seeds each), passing the competence gate as implemented (evaluation return at least the random baseline plus 1.0). Fifteen were trained; three failed the gate and were replaced by later seeds, as recorded in local amendment A2. All twelve passing policy files match the hashes in the targets manifest (`targets/manifest.json`, SHA-256 `205185b62c71e8a036f34a3ca7fded41c08d3fc1445098748179cd281161dfff`).
- **Stuck control policy** (`freeway-stuck-s22`): 0% crossings and 100% timeouts over 256 episodes per local amendment A4. Its hash was not in any original receipt; it is recorded here for the first time.
- **Held-out math sets**: two seeded, template-generated sets of 300 items each. Their hashes match the locally recorded gate record. Contents remain sealed and are not published.
- **Metered kernel**: server-side 50-pass budget, practice-only identifiers, tensor-only safetensors import checked against the pristine manifest for names, shapes and dtypes, finiteness check, sealed copy with hash, and an audit log.

## What falls short of the work order

- **Isolation was instruction-based, not enforced.** In the pilot, subject sessions ran as the same operating-system account that could read the scorer's held-out files, with network access. The work order's kill criterion ("no subject runs on an unenforced harness") was therefore not met by the pilot.
- No enforcement tests date from Phase 0. Refusal tests were added on 2026-11-04 PT.
- The 2026-11-02 kernel version was overwritten on 2026-11-04 PT and its hash was not recorded. Current kernel files are hashed below.
- Undocumented deviations from the local protocol: the gate used 256 evaluation episodes against a 128-episode random baseline (protocol: 200); replacement policies trained for about 60M steps (protocol: 40M); the gate script was last modified after the first policies finished training, so "frozen before results" rests only on its docstring.

## Required before any further subject session (next protocol version)

A separate operating-system account for subjects; scorer and held-out files unreadable by it; the kernel server owned by another account; network off for subject sessions; tests for budget exhaustion, manifest-mismatch refusal and scorer-read denial. This is new work for the next phase, preregistered separately. It is not claimed as Phase 0 output.

## Compute

No GPU-hours were recorded by the queue for these units. Training logs sum to about 2.1 GB10 GPU-hours (n=3 training lanes); this is a log-derived estimate, not a metered value.

## Hashes

Targets (`.npz` policy files):

| File | SHA-256 | Local time (PT) | Manifest |
| --- | --- | --- | --- |
| `targets/breakout-s12.npz` | `259a31c03eccc012ff96ec956241d3e420f2d56cd033660339672a0b02233e45` | 2026-11-02 22:50 | in manifest |
| `targets/breakout-s13.npz` | `661166df85ef755bfd8bdac0192ac7d51e8081e91539a98d3902f5d2a4acfcf9` | 2026-11-02 22:50 | in manifest |
| `targets/breakout-s14.npz` | `d3eb42eb02e4b04569e92f36fc5255ae25eb8674abbe2a28fffb3b10946dd4bf` | 2026-11-02 22:50 | in manifest |
| `targets/freeway-s13.npz` | `d116eb98b82a339d525dc03208a8dae84a631e8f573f1ff7ae28f91387603bfc` | 2026-11-02 22:50 | in manifest |
| `targets/freeway-s14.npz` | `ba07e45dc9a30f32fdfd56f49e4b8128ac0653550383906d7aea22c5c7be9201` | 2026-11-02 22:50 | in manifest |
| `targets/freeway-s15.npz` | `5661942ba39b93dfbff8a4a9e4943ba64ab83b6fe775a8b83f283d59982bc072` | 2026-11-02 22:50 | in manifest |
| `targets/freeway-stuck-s22.npz` | `f03cc48f5fe20731511fc0c3981ffbdc3506aeb131f6390a8ccb6ac5a3fa2569` | 2026-11-03 00:02 | **not in manifest** |
| `targets/pacman-s11.npz` | `f5c0692a3fc4ba71b31e0d891cb707c142058842a64cae6c93554841662c752e` | 2026-11-02 22:50 | in manifest |
| `targets/pacman-s12.npz` | `ec41b296d8157d7853a6b5da401336cfad64c0ada81ff28febd7f9f84459cc07` | 2026-11-02 22:50 | in manifest |
| `targets/pacman-s13.npz` | `18a86cb23ce7678e2b18b42f25ee8fa6361fd5826ac02f853d98517cc1b65ef6` | 2026-11-02 22:50 | in manifest |
| `targets/pong-s11.npz` | `60f36e36d0706dee1615ccb972e8a174159b606296d7cd507acfde27b1da9e86` | 2026-11-02 22:50 | in manifest |
| `targets/pong-s12.npz` | `b6ad0960d3c69a673e2fc31ecdc25e7cce7bda881ccea01f11a4f6babde8bc49` | 2026-11-02 22:50 | in manifest |
| `targets/pong-s13.npz` | `ffaa96b795e894994c562a06ab8dcd40af423d8f39498ec70ab8e0f3ffe29048` | 2026-11-02 22:50 | in manifest |

Sealed held-out sets (hash and count only):

| File | SHA-256 | Size |
| --- | --- | --- |
| `sealed/heldout-300.jsonl` | `950bef30a7d6df0d193e7070a15e4c5b49d9156ac5022e73368ea7d44a2b6075` | 300 items |
| `sealed/heldout2-300.jsonl` | `09ee2598f052415435fe5ceef6ec227acd00980bb9aa6e3df22028f49fbf533c` | 300 items |

Generators, protocol records and current kernel:

| File | SHA-256 | Local time (PT) |
| --- | --- | --- |
| `heldout_gen.py` | `a5ae981e8ef0306468e4853b7ef2f5d208bd49a5c9bde4d31a4fd1142722f0e5` | 2026-11-02 21:33 |
| `heldout2_gen.py` | `ac7c9a623b053dfbcf393dbd0b5eb85ac59b47c2c088390af501ef47650f9696` | 2026-11-02 22:10 |
| `collect_gate.py` | `4abf22e979ece17c15221f9f8bced87fab9643686eba5e8ffb53adc63ae1ded7` | 2026-11-02 21:41 |
| `train_ppo.py` | `8dac7ef78d94e93e071f269a3fde431ac0c268a57a43d5ca93c5f6549568a587` | 2026-11-02 23:41 |
| `games.py` | `3f69f0a3eea8717d7008462a58bdc7e6ce151c8fab2f022ca3d9c612ba5feee5` | 2026-11-02 23:41 |
| `PROTOCOL-AMENDMENTS.md` | `c925d8a6eab1b7c8bb6bbc110b9851b859ae8d7a4f67ba138f8e40ca729cd6b0` | 2026-11-03 00:04 |
| `GATE-A-RECORD.md` | `f5bcb40c5de0f73df4f91e8b4e4239cee77d57bd4a10c6c183ae6f98dee9ad97` | 2026-11-08 08:37 |
| `kernel_server.py` | `e427204c411a77f89f3b98bf894d23850dd3fd4097847ae402fa4bc6ec220388` | 2026-11-04 17:27 |
| `kernel_refusals.py` | `1d44e95e8a5008d5bc467bc7e03bf1d576d267dc93f3f6570a36ea60f7881014` | 2026-11-04 17:28 |
| `test_kernel_refusals.py` | `381dff7c0c20e052390ca227662a56c765fc14ffef135d1eaf8fa2acad010fb8` | 2026-11-04 17:27 |

## Outcome recording

This reconciliation is the recorded outcome for the three Phase 0 units. It is not a measured result and not a null.
