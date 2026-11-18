"""Check that a fresh run reproduces the results shipped with the package.

    python scripts/verify_replication.py            # full check: recompute everything (~1.5-2 h)
    python scripts/verify_replication.py --quick    # only rebuild tables/figures/numbers (seconds)

Steps:
1. The first time it runs, it copies the shipped outputs (results/, paper/tables,
   paper/numbers.tex) to reference_outputs/. They are never modified afterwards.
2. It deletes the generated outputs (the full check deletes results/ too; the raw data
   in data/raw are never touched).
3. It runs scripts/run_all.py.
4. It compares every new number, table and result file with the reference copy and
   writes a report to verification_report.csv.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "reference_outputs"
RESULTS = ROOT / "results"
PAPER = ROOT / "paper"


def snapshot():
    """Keep a copy of the shipped outputs to compare against (only done once)."""
    if REF.exists():
        print(f"Reference copy already exists: {REF.relative_to(ROOT)}/ (left unchanged)")
        return
    shutil.copytree(RESULTS, REF / "results")
    shutil.copytree(PAPER / "tables", REF / "tables")
    shutil.copy2(PAPER / "numbers.tex", REF / "numbers.tex")
    print(f"Saved the shipped outputs to {REF.relative_to(ROOT)}/")


def clean(full: bool):
    if full:
        shutil.rmtree(RESULTS, ignore_errors=True)
        RESULTS.mkdir()
    for d in (PAPER / "tables", PAPER / "figures"):
        shutil.rmtree(d, ignore_errors=True)
    (PAPER / "numbers.tex").unlink(missing_ok=True)
    print("Deleted generated outputs" + (" including results/" if full else " (tables, figures, numbers)"))


def run(full: bool):
    cmd = [sys.executable, "-u", str(ROOT / "scripts" / "run_all.py")]
    if not full:
        cmd += ["--stages", "report"]
    print("Running: python scripts/run_all.py" + ("" if full else " --stages report"), flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait() != 0:
        raise RuntimeError("run_all.py failed; see the messages above.")


def _macros(path: Path) -> dict:
    return dict(re.findall(r"\\newcommand\{\\(\w+)\}\{(.*)\}", path.read_text()))


def _to_float(s: str):
    try:
        return float(s.replace(",", "").replace("$<$", ""))
    except ValueError:
        return None


def compare(tol: float = 1e-6) -> pd.DataFrame:
    rows = []
    # 1. every number quoted in the text
    old, new = _macros(REF / "numbers.tex"), _macros(PAPER / "numbers.tex")
    for k in sorted(set(old) | set(new)):
        a, b = old.get(k), new.get(k)
        ok = a == b
        diff = None
        if not ok and a is not None and b is not None:
            fa, fb = _to_float(a), _to_float(b)
            if fa is not None and fb is not None:
                diff = abs(fa - fb)
        rows.append({"kind": "number in text", "item": k, "reference": a, "new": b,
                     "identical": ok, "abs_diff": diff})
    # 2. every LaTeX table
    for f in sorted((REF / "tables").glob("*.tex")):
        g = PAPER / "tables" / f.name
        same = g.exists() and f.read_text() == g.read_text()
        rows.append({"kind": "table", "item": f.name, "identical": same})
    # 3. every result file (numeric columns compared with a tolerance)
    for f in sorted((REF / "results").glob("*")):
        g = RESULTS / f.name
        if not g.exists():
            rows.append({"kind": "result file", "item": f.name, "identical": False, "new": "missing"})
            continue
        if f.suffix == ".json":
            same = f.read_text() == g.read_text()
            rows.append({"kind": "result file", "item": f.name, "identical": same})
            continue
        a, b = pd.read_csv(f), pd.read_csv(g)
        keys = [c for c in ("variant", "dgp", "model", "rep", "date", "refit_date", "loss") if c in a.columns]
        if keys and set(keys) <= set(b.columns):
            a = a.sort_values(keys, kind="stable").reset_index(drop=True)
            b = b.sort_values(keys, kind="stable").reset_index(drop=True)
        if a.shape != b.shape or list(a.columns) != list(b.columns):
            rows.append({"kind": "result file", "item": f.name, "identical": False,
                         "new": f"shape {b.shape} vs {a.shape}"})
            continue
        num = a.select_dtypes("number").columns
        maxdiff = float(np.nanmax(np.abs(a[num].to_numpy(float) - b[num].to_numpy(float)))) if len(num) else 0.0
        other = [c for c in a.columns if c not in num]
        same_text = all((a[c].astype(str) == b[c].astype(str)).all() for c in other)
        rows.append({"kind": "result file", "item": f.name, "identical": same_text and maxdiff <= tol,
                     "abs_diff": maxdiff})
    # 4. figures exist
    for name in ["fig_returns", "fig_pit", "fig_cumloss", "fig_var", "fig_parameters"]:
        rows.append({"kind": "figure", "item": name + ".pdf",
                     "identical": (PAPER / "figures" / f"{name}.pdf").exists()})
    rep = pd.DataFrame(rows)
    rep.to_csv(ROOT / "verification_report.csv", index=False)
    return rep


def summarize(rep: pd.DataFrame):
    print("\n=== Verification summary ===")
    for kind, g in rep.groupby("kind", sort=False):
        print(f"{kind:15s}: {int(g.identical.sum())} of {len(g)} identical")
    bad = rep[~rep.identical]
    if bad.empty:
        print("\nEverything matches the shipped results exactly.")
    else:
        print(f"\n{len(bad)} items differ (details in verification_report.csv):")
        print(bad.to_string(index=False, max_colwidth=60))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="only rebuild tables/figures/numbers")
    ap.add_argument("--compare-only", action="store_true", help="skip running; just compare")
    args = ap.parse_args()
    snapshot()
    if not args.compare_only:
        clean(full=not args.quick)
        run(full=not args.quick)
    summarize(compare())


if __name__ == "__main__":
    main()
