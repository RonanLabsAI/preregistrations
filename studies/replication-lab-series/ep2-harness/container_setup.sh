#!/usr/bin/env bash
# Build the two Python environments inside the pinned vLLM image (run once, idempotent).
# venv-train: the CIG-SFT fork with the paper's library pins, reusing the image's sm_120 torch.
# venv-eval:  the image's vLLM plus math-verify.
set -euo pipefail
W=/work
python3 -c 'import torch; print(torch.__version__)' > "$W/pydeps/torch-version.txt"
printf 'torch==%s\n' "$(cat "$W/pydeps/torch-version.txt")" > "$W/pydeps/constraints.txt"
if [ ! -x "$W/venv-train/bin/python" ]; then
  python3 -m venv --system-site-packages "$W/venv-train"
  "$W/venv-train/bin/pip" install -q -c "$W/pydeps/constraints.txt" \
    transformers==4.57.1 datasets==4.0.0 accelerate==1.11.0 peft==0.18.1 trl==0.24.0 \
    einops fire omegaconf "tyro<0.9.0" hf-transfer protobuf pyyaml pydantic scipy pandas matplotlib
  "$W/venv-train/bin/pip" install -q --no-deps -e "$W/CIG-SFT-public"
fi
if [ ! -x "$W/venv-eval/bin/python" ]; then
  python3 -m venv --system-site-packages "$W/venv-eval"
  "$W/venv-eval/bin/pip" install -q -c "$W/pydeps/constraints.txt" math-verify==0.9.0
fi
"$W/venv-train/bin/python" - <<'EOF' > "$W/pydeps/train-env.json"
import json, importlib
mods = ["torch", "transformers", "datasets", "accelerate", "peft", "trl", "llamafactory"]
print(json.dumps({m: getattr(importlib.import_module(m), "__version__", "?") for m in mods}))
EOF
"$W/venv-eval/bin/python" - <<'EOF' > "$W/pydeps/eval-env.json"
import json, importlib, importlib.metadata as md
print(json.dumps({"vllm": importlib.import_module("vllm").__version__, "torch": importlib.import_module("torch").__version__,
                  "transformers": importlib.import_module("transformers").__version__, "math-verify": md.version("math-verify")}))
EOF
cat "$W/pydeps/train-env.json" "$W/pydeps/eval-env.json"
