"""Native K3 result-package writer shared by the W16b study harnesses.

Writes manifest.json, receipts/, figures/, drafts/ and VERDICT.md into the
directory Mission Control exports as RL_RECEIPT_DIR. Identity (product, run,
arm, seed, fingerprint) must match the completed unit exactly; the harness
refuses to run when the observed environment fingerprint differs from the
frozen one. Standard library only.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import platform
from pathlib import Path
from zoneinfo import ZoneInfo

PT = ZoneInfo("America/Los_Angeles")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def now_pt() -> str:
    return dt.datetime.now(PT).isoformat(timespec="seconds")


class FingerprintMismatch(RuntimeError):
    pass


def check_fingerprint(observed: str, expected: str | None) -> None:
    """Fail closed when the frozen fingerprint is set and differs."""
    if expected is not None and expected != observed:
        raise FingerprintMismatch(f"environment fingerprint drift: expected {expected!r}, observed {observed!r}")


def identity_from_env(default_product: str) -> dict:
    """Mission Control exports RL_PRODUCT_ID, RL_UNIT_ID, RL_RUN_ID, RL_RECEIPT_DIR."""
    return {
        "product_id": os.environ.get("RL_PRODUCT_ID", default_product),
        "unit_id": os.environ.get("RL_UNIT_ID", "local"),
        "run_id": os.environ.get("RL_RUN_ID", "local-run"),
        "receipt_dir": os.environ.get("RL_RECEIPT_DIR"),
    }


def host_summary() -> dict:
    return {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "system": platform.system(),
        "cpus": os.cpu_count(),
    }


class Package:
    """Builds one native K3 package; never overwrites an existing one."""

    def __init__(self, root: Path):
        self.root = Path(root)
        if self.root.exists() and any(self.root.iterdir()):
            raise FileExistsError(f"receipt directory is not empty: {self.root}")
        for sub in ("receipts", "figures", "drafts"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self.receipts: list[str] = []
        self.frozen: list[dict] = []
        self._log = open(self.root / "receipts" / "run.log", "a", encoding="utf-8")
        self.receipts.append("receipts/run.log")

    def log(self, event: str, **fields) -> None:
        row = {"t": now_pt(), "event": event, **fields}
        self._log.write(json.dumps(row, sort_keys=True) + "\n")
        self._log.flush()

    def add_receipt(self, relative: str, data: bytes | str) -> str:
        if not relative.startswith("receipts/") or ".." in relative.split("/"):
            raise ValueError("receipts must live under receipts/")
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "xb") as f:
            f.write(data.encode() if isinstance(data, str) else data)
        if relative not in self.receipts:
            self.receipts.append(relative)
        return relative

    def add_frozen_input(self, relative: str, source: Path, expected_sha256: str) -> None:
        data = Path(source).read_bytes()
        digest = sha256_bytes(data)
        if digest != expected_sha256:
            raise ValueError(f"frozen input hash mismatch for {relative}: {digest}")
        self.add_receipt(relative, data)
        self.frozen.append({"path": relative, "sha256": digest})

    def write_jsonl(self, relative: str, rows: list[dict]) -> str:
        return self.add_receipt(relative, "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows))

    def finish(self, *, identity: dict, title: str, lane: str, prereg_path: Path, prereg_url: str,
               arm: str, seed: int, environment: dict, verdict: str, tiers: list[str]) -> Path:
        if not self.frozen:
            raise ValueError("at least one frozen input is required")
        prereg_sha = sha256_file(prereg_path)
        self.add_receipt("receipts/preregistration.md", Path(prereg_path).read_bytes())
        self._log.close()
        manifest = {
            "schema_version": "1.0",
            "product_id": identity["product_id"],
            "run_id": identity["run_id"],
            "title": title,
            "lane": lane,
            "created_at": now_pt(),
            "preregistration": {"sha256": prereg_sha, "url": prereg_url, "timestamp_verifiable": True},
            "backlog": False,
            "arms": [arm],
            "seeds": [seed],
            "frozen_inputs": self.frozen,
            "environment": environment,
            "receipts": self.receipts,
            "figures": [],
            "publication_tiers": tiers,
        }
        (self.root / "VERDICT.md").write_text(verdict, encoding="utf-8")
        (self.root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return self.root
