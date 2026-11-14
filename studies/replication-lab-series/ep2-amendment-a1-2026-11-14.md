# Replication lab episode 2: amendment A1, harness freeze (2026-11-14 PT)

Amends `ep2-protocol-v1-2026-11-14.md` (commit `58f2aa354cf6d252b5f94b143afaf4690945d4bc`). This is the harness amendment that protocol v1 requires before the smoke run. Nothing has been trained or evaluated. Everything not listed here is unchanged and binding.

## Frozen inputs

- Code and data: `https://github.com/NiuKequan/CIG-SFT-public` commit `b3a2f7ffa3e8ea474158dd4950f06be29a1ad543`.
  - recipe `example/configs/qwen2_5_math_1p5b_full_cig_sft.yaml`: SHA-256 `a7c267b3e62cfba897fb741aab2329d6e4ebf5122301645debaf83de17b0f6fc`
  - training data `example/data/numina_cot/30k/train.parquet`: SHA-256 `4790bd8ccf7634332bda4aaf56593660805323946df4c6df3bf0938ac56c77f1`
  - CIG loss implementation `src/llamafactory/train/trainer_utils.py`: SHA-256 `2358431b734db92fba542d0d8d12ec1825b6ee6b267eccaa191f32ea1c60e2ed`
- Base model: `Qwen/Qwen2.5-Math-1.5B` revision `4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2`; `model.safetensors` SHA-256 `f7aa8d665ab44d3b461bed742fe93cdcedc15d2e31e50dc372c126e6f720857b`.
- Evaluation set: `HuggingFaceH4/MATH-500` revision `6e4ed1a2a79af7d8630a6b768ec859cb5af4d3be`, `test.jsonl` SHA-256 `35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132` (500 problems).
- Rendered evaluation prompts (token ids from the fork's own `qwen` template, user turn = problem + the training data's boxed-answer suffix): SHA-256 `466edbdf3a73ae4a62c86c33dab1dcd295a4072e2c1f3f1b6f82c330d52bc70a`.
- Container image: `vllm/vllm-openai:nightly-3f1bcfd82bb2e93fcd364fc7a5491718453a77e2` (`sha256:cb9bc64f0f038c2c2ebf13b31248fb87aca40c356c3d20f363be087cfcff1fcd`); PyTorch 2.14.0 with CUDA 13.0 (sm_120). Training environment pins the paper's libraries where the GPU allows: transformers 4.57.1, datasets 4.0.0, accelerate 1.11.0, peft 0.18.1, trl 0.24.0, the fork installed from the commit above. Evaluation: vLLM 0.32.0rc1 (dev190) and math-verify 0.9.0. All file hashes are in `harness/inputs.lock.json`.

## Harness and fingerprint

Code: `harness/` (`run_unit.py`, `train_arm.py`, `eval_math500.py`, `render_prompts.py`, `container_setup.sh`, `inputs.lock.json`, `common/k3.py`). Frozen environment fingerprint: `rtxpro6000-sm120|image@cb9bc64f0f03|torch2.14.0+cu130|tf4.57.1|trl0.24.0|vllm0.32.0rc1.dev190+g3f1bcfd82|mathverify0.9.0|cig@b3a2f7f|code@6628d6ce1ded`.

## Clarifications fixed before any result

1. Generation stops at the template's turn end (`<|im_end|>`) or `<|endoftext|>` for every arm, including the untrained base model, within the 4,096-token cap.
2. Training differs from the published recipe only in paths, `use_cig_sft_loss: false` for the SFT arm (and removal of the CIG-only keys), `seed` and `data_seed` 42, checkpoints every 25 steps with optimizer state (`save_only_model: false`, two kept), and `overwrite_output_dir: false` so a preempted run resumes from its latest checkpoint.
3. The CIG credit cache is computed once from the frozen base model and data and reused by the CIG arm; the smoke builds it.
4. Per-problem sampling seeds are `42 * 100003 + problem index`.

Smoke follows public commit of this amendment.

