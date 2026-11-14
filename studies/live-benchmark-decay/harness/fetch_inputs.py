"""Fetch the pinned public inputs and build the derived LiveBench release index.

Usage: python fetch_inputs.py <dest-dir> [--write-lock]
Without --write-lock it verifies every hash against inputs.lock.json and fails on drift.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
LCB_REV = "f186ab5041d6b768733b8ae8145151c867942134"
LCB_URL = f"https://raw.githubusercontent.com/LiveCodeBench/livecodebench.github.io/{LCB_REV}/build/performances_generation.json"
LB_JUDGMENT = ("model_judgment", "9704e5da7bfbefe75ac1482a13de827127295993", "data/leaderboard-00000-of-00001.parquet")
LB_QUESTIONS = {
    "reasoning": "6fc6498a5dfba553f69f4413feabade1f1a2d384",
    "math": "bb66571c8ccf32d3df9e6f48b920d3770ff4aacb",
    "coding": "a958549fdd8aa57be0a3fafe7b205ffc160ed5f4",
    "language": "3ada32a2e53d5e04e57fa503384cb85ce9116c40",
    "data_analysis": "31b9661ff678df9958e2f7fa228427f4c858c1a1",
    "instruction_following": "0868379c4b5cf62aeacaf8be4f08fced815c81bb",
}


def hf_url(name: str, rev: str, path: str) -> str:
    return f"https://huggingface.co/datasets/livebench/{name}/resolve/{rev}/{path}"


def get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ronanlabs-prereg-fetch/1"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return r.read()


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main() -> int:
    import pyarrow.parquet as pq

    dest = Path(sys.argv[1])
    write_lock = "--write-lock" in sys.argv
    dest.mkdir(parents=True, exist_ok=True)
    lock = {"lcb": {"url": LCB_URL, "revision": LCB_REV}, "livebench": {}, "frozen_files": {}}
    lcb = get(LCB_URL)
    (dest / "lcb_performances_generation.json").write_bytes(lcb)
    lock["lcb"]["sha256"] = sha(lcb)
    name, rev, path = LB_JUDGMENT
    mj = get(hf_url(name, rev, path))
    lock["livebench"]["model_judgment"] = {"url": hf_url(name, rev, path), "revision": rev, "sha256": sha(mj)}
    # Packages accept text receipts only: freeze a deterministic CSV of the pinned parquet.
    jcols = ["question_id", "model", "task", "category", "turn", "score"]
    jrows = pq.read_table(io.BytesIO(mj), columns=jcols).to_pylist()
    jrows.sort(key=lambda r: (r["question_id"], r["model"], int(r["turn"])))
    jbuf = io.StringIO()
    jw = csv.writer(jbuf, lineterminator="\n")
    jw.writerow(jcols)
    for r in jrows:
        jw.writerow([r["question_id"], r["model"], r["task"], r["category"], int(r["turn"]), repr(float(r["score"]))])
    (dest / "livebench_model_judgment.csv").write_bytes(jbuf.getvalue().encode())
    rows = []
    for cat, qrev in LB_QUESTIONS.items():
        url = hf_url(cat, qrev, "data/test-00000-of-00001.parquet")
        data = get(url)
        lock["livebench"][cat] = {"url": url, "revision": qrev, "sha256": sha(data)}
        cols = ["question_id", "category", "task", "livebench_release_date"]
        table = pq.read_table(io.BytesIO(data), columns=cols).to_pylist()
        for r in table:
            rows.append({c: str(r[c]) for c in cols})
    rows.sort(key=lambda r: r["question_id"])
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=["question_id", "category", "task", "livebench_release_date"], lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    index = buf.getvalue().encode()
    (dest / "livebench_release_index.csv").write_bytes(index)
    for fname in ("lcb_performances_generation.json", "livebench_model_judgment.csv", "livebench_release_index.csv"):
        lock["frozen_files"][fname] = {"sha256": sha((dest / fname).read_bytes())}
    if write_lock:
        (HERE / "inputs.lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
        print("lock written")
        return 0
    frozen = json.loads((HERE / "inputs.lock.json").read_text())
    if frozen != lock:
        print("DRIFT: fetched inputs differ from inputs.lock.json", file=sys.stderr)
        return 1
    print("inputs verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
