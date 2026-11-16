#!/usr/bin/env python3
# Phase 2 sequencer: a block starts only if its whole A-B-B-A cost fits before deadline (claim + 6.0h - 0.25h reserve).
import json, subprocess, sys, time
from datetime import datetime
from zoneinfo import ZoneInfo
for b in ("b1", "b2", "b3"):
    p = json.load(open("plan-p2-%s.json" % b))
    now = time.time()
    row = dict(block=b, cost_minutes=p["block_cost_minutes"], now_pt=datetime.now(ZoneInfo("America/Los_Angeles")).isoformat())
    if now + p["block_cost_minutes"] * 60 > p["deadline_epoch"]:
        row.update(disposition="unrun", reason="budget: whole block does not fit before the restore reserve")
        print(json.dumps(row), flush=True)
        with open("loads.jsonl", "a") as f: f.write(json.dumps(row) + "\n")
        continue
    rc = subprocess.run([sys.executable, "run_loads.py", "plan-p2-%s.json" % b]).returncode
    row.update(disposition="ran", rc=rc)
    print(json.dumps(row), flush=True)
    if rc != 0:
        print(json.dumps(dict(event="halt", block=b, rc=rc)), flush=True)
        break
