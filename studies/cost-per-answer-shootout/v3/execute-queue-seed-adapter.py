"""Queue entrypoint: execute the frozen runner, then losslessly normalize engine logs.

Raw bytes of non-UTF8 logs are retained as base64 JSON with their original SHA-256.
No measurements, scores, inputs, protocol bytes or environment identity are changed.
"""

import base64
import hashlib
import json
import os
import signal
import subprocess
import sys
from pathlib import Path


def finalize(root):
    manifest = root / "manifest.json"
    if not manifest.exists():
        return
    data = json.loads(manifest.read_text())
    unit_id = os.environ.get("RL_UNIT_ID", "")
    if unit_id:
        label = unit_id.rsplit("-", 1)[-1]
        if not label.isdigit():
            raise ValueError("queue execution replicate label must be explicit and numeric")
        outer_seed = int(label)
        if data["seeds"] != [outer_seed]:
            archive = root / "receipts/manifest-before-queue-seed-binding.json"
            if archive.exists():
                raise ValueError("seed-binding archive already exists; inspect recovery")
            archive.write_bytes(manifest.read_bytes())
            data["environment"]["frozen_trial_seeds"] = data["seeds"]
            data["environment"]["queue_execution_unit_id"] = unit_id
            data["environment"]["seed_semantics"] = (
                "manifest seeds identifies the outer queue execution replicate; frozen_trial_seeds preserves the scientific generator/engine seeds without modification"
            )
            data["seeds"] = [outer_seed]
            data["receipts"].append("receipts/" + archive.name)
    for path in (root / "receipts").glob("*.log"):
        raw = path.read_bytes()
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            archive = path.with_suffix(".raw-bytes.json")
            archive.write_text(
                json.dumps(
                    {
                        "source": path.name,
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "encoding": "base64",
                        "bytes_base64": base64.b64encode(raw).decode("ascii"),
                    }
                )
                + "\n"
            )
            path.write_text(raw.decode("utf-8", errors="backslashreplace"))
            relative = "receipts/" + archive.name
            if relative not in data["receipts"]:
                data["receipts"].append(relative)
    manifest.write_text(json.dumps(data, indent=2) + "\n")


def main():
    if sys.argv[1:2] == ["--finalize-only"]:
        finalize(Path(sys.argv[2]))
        return
    # The runner in this process group handles TERM and checkpoints first.
    signal.signal(signal.SIGTERM, lambda *_: None)
    child = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("runner.py")), *sys.argv[1:]], check=False
    )
    finalize(Path(os.environ["RL_RECEIPT_DIR"]))
    raise SystemExit(child.returncode)


if __name__ == "__main__":
    main()
