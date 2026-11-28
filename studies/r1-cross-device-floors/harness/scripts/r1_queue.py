"""R1 per-device queue: runs this device's plan runs in plan order over ssh, one at a time.

usage: R1_HOST_<DEV>=host R1_GPU_UUID_<DEV>=uuid python3 scripts/r1/r1_queue.py --device rtx6000 [--arm transfer ...]
Skips a run whose results/r1/<run_id>/run_meta.json exists and has no "aborted" key; an aborted
run dir is renamed to <run_id>.aborted-<UTC timestamp> and rerun once (a second abort stops the queue);
stops at the first failure
(card: any run whose engine config differs is discarded and rerun, decided by the analysis, not here). Requires an
active fleet claim whose id starts with mom-r1-. Events: logs/r1/<device>.events.log (UTC).
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch as L  # noqa: E402

def claim_held() -> bool:
    out = subprocess.run(["fleet-claims", "active"], capture_output=True, text=True).stdout
    try:
        items = json.loads(out)
    except Exception:
        return False
    items = list(items.values()) if isinstance(items, dict) else items
    return any(str(c.get("id", "")).startswith("mom-r1-") and not c.get("released") for c in items)

OFFLINE_ARMS = ("transfer", "matched_rane", "full_set", "warmup")  # the HTTP screen, adapter, cost and logprob arms have
# their own runners (r1_http.py, r1_cost.py; adapter LoRA serving and the item limit are not in r1_offline yet)


def meta_is_clean(meta) -> bool:
    """Done only when run_meta.json exists and carries no 'aborted' key."""
    return isinstance(meta, dict) and "aborted" not in meta


def decide_existing(meta, retried: bool) -> str:
    if meta is None:
        return "run"
    if meta_is_clean(meta):
        return "skip"
    return "stop" if retried else "archive_and_rerun"


def decide_after_run(meta, retried: bool) -> str:
    if meta is None:
        return "fail"
    if meta_is_clean(meta):
        return "finish"
    return "stop" if retried else "archive_and_rerun"


def aborted_dir_name(run_id: str, timestamp: str = None) -> str:
    ts = timestamp or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    return f"{run_id}.aborted-{ts}"


def fetch_run_meta(host: str, run_id: str):
    """run_meta.json dict via ssh cat, or None when missing/unreadable."""
    remote = f"cat $HOME/mom-lab/r1/results/r1/{run_id}/run_meta.json"
    r = subprocess.run(["ssh", host, remote], capture_output=True, text=True)
    if r.returncode != 0 or not (r.stdout or "").strip():
        return None
    try:
        return json.loads(r.stdout)
    except Exception:
        return None


def archive_aborted_run(host: str, run_id: str) -> str:
    """Rename the aborted run dir aside on the device; returns the archive name."""
    name = aborted_dir_name(run_id)
    base = "$HOME/mom-lab/r1/results/r1"
    subprocess.run(["ssh", host, f"mv {base}/{run_id} {base}/{name}"], capture_output=True, text=True)
    return name


def image_digest_cmd(image: str) -> list[str]:
    """Host command that resolves the pinned image's repo digest (C3)."""
    return ["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", image]

def fetch_image_digest(host: str, image: str = None) -> str:
    """Repo digest of `image` on the device host, via ssh. Raises on failure/empty."""
    image = image or L.PIN_IMAGE
    remote = L.shell_cmd(image_digest_cmd(image))  # quoted: the Go template has spaces
    r = subprocess.run(["ssh", host, remote], capture_output=True, text=True)
    digest = (r.stdout or "").strip().splitlines()
    digest = digest[0].strip() if digest else ""
    if r.returncode != 0 or not digest:
        raise RuntimeError(f"could not resolve image digest for {image} on {host} (rc={r.returncode})")
    return digest

def select_runs(plan: dict, device: str, arms=None) -> list[dict]:
    want = tuple(arms) if arms else OFFLINE_ARMS
    bad = [x for x in want if x not in OFFLINE_ARMS]
    if bad:
        raise SystemExit(f"queue.py runs offline arms only {OFFLINE_ARMS}; not {bad}")
    return [r for r in plan["runs"] if r["device"] == device and r["arm"] in want]

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", required=True); ap.add_argument("--arm", action="append")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    host = os.environ[f"R1_HOST_{a.device.upper()}"]
    runs = select_runs(L.build_plan(), a.device, a.arm)
    ev = ROOT / "logs/r1" / f"{a.device}.events.log"; ev.parent.mkdir(parents=True, exist_ok=True)
    def log(msg):
        line = time.strftime("%Y-%m-%d %H:%M:%S UTC ", time.gmtime()) + msg
        print(line, flush=True); ev.open("a").write(line + "\n")
    log(f"QUEUE {a.device} {len(runs)} runs arms={a.arm or 'all offline'}")
    for r in runs:
        retried = False
        while True:
            existing = fetch_run_meta(host, r["run_id"]) if not a.dry_run else None
            action = decide_existing(existing, retried) if not a.dry_run else "run"
            if action == "skip":
                log(f"SKIP {r['run_id']} (done)"); break
            if action == "stop":
                log(f"STOP {r['run_id']}: second aborted run; queue stopped"); return 1
            if action == "archive_and_rerun":
                name = archive_aborted_run(host, r["run_id"])
                log(f"ABORTED {r['run_id']} moved aside to {name}; rerunning once")
                retried = True
            if not claim_held():
                log("STOP: no active mom-r1-* fleet claim"); return 2
            if a.dry_run:
                cmd = L.shell_cmd(L.docker_run_cmd(a.device, L.offline_inner(r)))
                print(cmd); break
            try:
                digest = fetch_image_digest(host)  # before each run, on the device host
            except Exception as e:
                log(f"FAIL {r['run_id']} image digest: {e}"); return 1
            cmd = L.shell_cmd(L.docker_run_cmd(a.device, L.offline_inner(r), image_digest=digest))
            log(f"START {r['run_id']}"); t0 = time.time()
            rc = subprocess.run(["ssh", host, cmd]).returncode
            after = fetch_run_meta(host, r["run_id"])
            verdict = decide_after_run(after, retried)
            if verdict == "finish" and rc == 0:
                log(f"FINISH {r['run_id']} {time.time() - t0:.0f} s"); break
            if verdict == "archive_and_rerun":
                name = archive_aborted_run(host, r["run_id"])
                log(f"ABORTED {r['run_id']} moved aside to {name}; rerunning once")
                retried = True; continue
            if verdict == "stop":
                log(f"STOP {r['run_id']}: second aborted run; queue stopped"); return 1
            log(f"FAIL {r['run_id']} rc={rc}"); return 1
    log("QUEUE_EMPTY"); return 0

if __name__ == "__main__":
    sys.exit(main())
