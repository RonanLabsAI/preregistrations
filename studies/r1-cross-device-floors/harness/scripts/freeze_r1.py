"""Write/extend artifacts/R1_FREEZE.sha256 for R1 (cross-device noise floors).

Based on scripts/addendum/freeze_addendum.py: same manifest format, one
"sha256  relative/path" line per frozen file.

usage: python3 scripts/r1/freeze_r1.py --stage {card,code} | --fix N --reason TEXT

--stage card freezes preregs/2026-11-28-r1-floors.md, every
scripts/r1/*.py and configs/r1/* file present, and tests/test_r1_*.py.
Later stages append the new/changed files from that same set under a
"# stage <name> <PT timestamp>" header. A stage refuses (exit 1) once
its outputs exist: any results/r1* exists. freeze_ok(stage) verifies every manifest line
up to and including that stage, for the launcher.

The repo root is overridable via the R1_ROOT env var or a root argument,
so tests can run against a tmp dir.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

STAGES = ["card", "code"]
CARD = "preregs/2026-11-28-r1-floors.md"
PATTERNS = [CARD, "scripts/r1/*.py", "scripts/r1/*.sh", "configs/r1/**/*", "tests/test_r1_*.py", "docker/Dockerfile*"]
EXCLUDE_PARTS = {"__pycache__"}
OUT_REL = "artifacts/R1_FREEZE.sha256"


def _root(root=None) -> Path:
    if root is not None:
        return Path(root)
    env = os.environ.get("R1_ROOT")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2]


def files(root=None) -> list[Path]:
    wt = _root(root)
    out = set()
    for pat in PATTERNS:
        for p in wt.glob(pat):
            if p.is_file() and not (EXCLUDE_PARTS & set(p.parts)):
                out.add(p)
    return sorted(out, key=lambda p: p.relative_to(wt).as_posix())


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def _entries(root) -> list[tuple[str, str]]:
    wt = _root(root)
    return [(sha(p), p.relative_to(wt).as_posix()) for p in files(wt)]


def _header(stage: str) -> str:
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    return f"# stage {stage} {now:%Y-%m-%d %H:%M:%S} {now.tzname()}\n"


def _refused(stage: str, root) -> list[Path]:
    results = _root(root) / "results"
    if not results.is_dir():
        return []
    pats = ["r1*"]  # R1 writes results/r1*/ only; any output blocks every stage
    return sorted({p for pat in pats for p in results.glob(pat)})


def _parse(text: str):
    """Yield (stage, digest, rel) for each manifest entry, in file order."""
    stage = None
    for line in text.splitlines():
        if line.startswith("# stage "):
            parts = line.split()
            stage = parts[2] if len(parts) > 2 else None
        elif line.strip():
            digest, _, rel = line.partition("  ")
            yield stage, digest.strip(), rel.strip()


def _sections(text: str):
    """Yield (stage, 'YYYY-MM-DD HH:MM:SS', [(digest, rel)]) per '# stage' header, in file order."""
    cur = None
    for line in text.splitlines():
        if line.startswith("# stage "):
            parts = line.split()
            cur = [parts[2] if len(parts) > 2 else None, " ".join(parts[3:5]), []]
            yield cur
        elif line.strip() and cur is not None:
            digest, _, rel = line.partition("  ")
            cur[2].append((digest.strip(), rel.strip()))


def freeze_ok(stage: str, root=None) -> bool:
    """True iff `stage` has been frozen and every frozen file matches its latest frozen hash.

    All manifest stages and every logged fix overlay (R1_FREEZE_fix<N>.sha256) are applied in time order,
    so a later hash of a file supersedes an earlier one whichever file it is in (fix4: fixes applied last used to
    override newer stage hashes).
    """
    if stage not in STAGES:
        return False
    wt = _root(root)
    out = wt / OUT_REL
    if not out.exists():
        return False
    secs = [s for s in _sections(out.read_text(encoding="utf-8")) if s[0] in STAGES]
    if not any(s[0] == stage for s in secs):  # frozen once its header exists, even with no new files
        return False
    for q in _fix_paths(wt):
        secs += list(_sections(q.read_text(encoding="utf-8")))
    latest: dict[str, str] = {}
    for _, _, entries in sorted(secs, key=lambda s: s[1]):  # stable: same-second order kept
        for digest, rel in entries:
            latest[rel] = digest
    if not latest:
        return False
    for rel, digest in latest.items():
        p = wt / rel
        if not p.is_file() or sha(p) != digest:
            return False
    return True


def _fix_paths(wt: Path) -> list[Path]:
    return sorted((wt / "artifacts").glob("R1_FREEZE_fix*.sha256"), key=lambda q: int(q.stem.split("fix")[1]))


def _fix_entries(wt: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for q in _fix_paths(wt):
        for _, digest, rel in _parse(q.read_text(encoding="utf-8")):
            out[rel] = digest
    return out


def freeze_fix(n: int, reason: str, root=None) -> int:
    """Deviation fix after output exists (paper 3's ADDENDUM_FREEZE_fix1 practice): hash every frozen-set file whose
    bytes differ from the latest frozen hash into artifacts/R1_FREEZE_fix<n>.sha256. Never rewrites the manifest."""
    wt = _root(root)
    out = wt / OUT_REL
    target = wt / "artifacts" / f"R1_FREEZE_fix{n}.sha256"
    if not out.exists() or target.exists() or not reason.strip():
        print("need an existing manifest, a new fix number and a reason", file=sys.stderr)
        return 1
    latest = {rel: d for _, d, rel in _parse(out.read_text(encoding="utf-8"))}
    latest.update(_fix_entries(wt))
    changed = [(d, r) for d, r in _entries(wt) if latest.get(r) != d]
    if not changed:
        print("nothing changed", file=sys.stderr)
        return 1
    now = datetime.now(ZoneInfo("America/Los_Angeles"))
    target.write_text(f"# stage fix{n} {now:%Y-%m-%d %H:%M:%S} {now.tzname()} reason: {reason.strip()}\n"
                      + "".join(f"{d}  {r}\n" for d, r in changed), encoding="utf-8")
    print(f"wrote {target.relative_to(wt)}: {len(changed)} files")
    return 0


def freeze(stage: str, root=None) -> int:
    wt = _root(root)
    present = _refused(stage, wt)
    if present:
        names = ", ".join(str(p.relative_to(wt)) for p in present)
        print(f"refusing to freeze stage {stage}: results already exist: {names}", file=sys.stderr)
        return 1
    if stage == "card" and not (wt / CARD).exists():
        print("card missing", file=sys.stderr)
        return 1
    entries = _entries(wt)
    out = wt / OUT_REL
    out.parent.mkdir(parents=True, exist_ok=True)
    if stage == "card" or not out.exists():
        text = _header(stage) + "".join(f"{d}  {r}\n" for d, r in entries)
        out.write_text(text, encoding="utf-8")
        print(f"wrote {OUT_REL}: stage {stage}, {len(entries)} files")
        return 0
    latest = {rel: d for _, d, rel in _parse(out.read_text(encoding="utf-8"))}
    new = [(d, r) for d, r in entries if latest.get(r) != d]
    with out.open("a", encoding="utf-8") as f:
        f.write(_header(stage))
        f.write("".join(f"{d}  {r}\n" for d, r in new))
    print(f"extended {OUT_REL}: stage {stage}, {len(new)} new/changed files")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=STAGES)
    ap.add_argument("--fix", type=int, help="write artifacts/R1_FREEZE_fix<N>.sha256 (deviation after output)")
    ap.add_argument("--reason", default="")
    a = ap.parse_args(argv)
    if a.fix is not None:
        return freeze_fix(a.fix, a.reason)
    if not a.stage:
        ap.error("--stage or --fix is required")
    return freeze(a.stage)


if __name__ == "__main__":
    sys.exit(main())
