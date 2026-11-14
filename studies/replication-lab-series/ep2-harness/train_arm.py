"""Train one arm with the authors' recipe (run in venv-train inside the container).

Only these keys differ from the published recipe: paths (model, data, outputs, credit
cache), `use_cig_sft_loss` (false for the SFT arm), checkpointing for preemption
(`save_steps`, `save_only_model: false`, `save_total_limit`, `overwrite_output_dir: false`)
and, for smoke only, `max_steps`. Resumes from the latest checkpoint when present.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path("/work/CIG-SFT-public")
RECIPE = REPO / "example/configs/qwen2_5_math_1p5b_full_cig_sft.yaml"


def build_config(arm: str, out_dir: Path, seed: int, save_steps: int, max_steps: int | None) -> dict:
    cfg = yaml.safe_load(RECIPE.read_text())
    cfg.update({
        "dataset_dir": str(REPO / "example/data"),
        "output_dir": str(out_dir),
        "overwrite_output_dir": False,
        "save_steps": save_steps,
        "save_only_model": False,
        "save_total_limit": 2,
        "seed": seed,
        "data_seed": seed,
        "use_cig_sft_loss": arm == "cig",
        "cig_credit_cache": "/work/runs/cig-cache/qwen2_5_math_1p5b_numina_cot_30k.pt",
    })
    cfg["model_path" if "model_path" in cfg else "model_name_or_path"] = "/work/models/Qwen2.5-Math-1.5B"
    if arm != "cig":
        for key in ("cig_lambda", "cig_erased_prompt", "cig_credit_cache"):
            cfg.pop(key, None)
    if max_steps:
        cfg["max_steps"] = max_steps
    return cfg


def latest_checkpoint(out_dir: Path) -> str | None:
    ckpts = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[1]))
    return str(ckpts[-1]) if ckpts else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["sft", "cig"], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--save-steps", type=int, default=25)
    ap.add_argument("--max-steps", type=int)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    cfg = build_config(args.arm, args.out, args.seed, args.save_steps, args.max_steps)
    resume = latest_checkpoint(args.out)
    if resume:
        cfg["resume_from_checkpoint"] = resume
    cfg_path = args.out / ("train-config-resume.yaml" if resume else "train-config.yaml")
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=True))
    print(json.dumps({"event": "train-start", "arm": args.arm, "resume": resume, "config": str(cfg_path)}), flush=True)
    rc = subprocess.call([sys.executable, str(REPO / "src/train.py"), str(cfg_path)], cwd=str(REPO))
    print(json.dumps({"event": "train-exit", "rc": rc}), flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
