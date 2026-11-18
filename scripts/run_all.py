"""Reproduce every number, table and figure in the article.

    python scripts/run_all.py                 # all stages, then tables and figures
    python scripts/run_all.py --stages forecasts report

Stages are cached: a stage whose outputs already exist in results/ is skipped
unless --force is given. Typical run time on a 2-core laptop: about 1.5 hours.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mxntail import pipeline, reporting  # noqa: E402

# Stages whose outputs are checked as a whole; "forecasts" and "robustness"
# instead reuse every model already stored and compute only missing ones.
OUTPUTS = {
    "data": ["diagnostics.json"],
    "insample": ["insample_fits.csv"],
    "montecarlo": ["montecarlo_scores.csv"],
}
ORDER = ["data", "insample", "forecasts", "montecarlo", "robustness", "report"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=ORDER, choices=ORDER)
    ap.add_argument("--force", action="store_true", help="recompute cached stages")
    args = ap.parse_args()
    cfg = pipeline.load_config()
    res = pipeline.results_dir()
    for stage in [s for s in ORDER if s in args.stages]:
        t0 = time.time()
        if stage == "report":
            reporting.build_all(cfg)
        elif stage in ("forecasts", "robustness"):
            pipeline.STAGES[stage](cfg, force=args.force)
        elif not args.force and all((res / f).exists() for f in OUTPUTS[stage]):
            print(f"[{stage}] cached, skipping (use --force to recompute)", flush=True)
            continue
        else:
            pipeline.STAGES[stage](cfg)
        print(f"[{stage}] done in {time.time() - t0:,.0f} s", flush=True)


if __name__ == "__main__":
    main()
