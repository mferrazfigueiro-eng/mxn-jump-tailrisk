"""Computation stages. Each stage reads config.toml and writes to results/.

Stages (run in order by scripts/run_all.py, or one by one from the notebooks):
    data         clean returns and descriptive diagnostics
    insample     full-sample maximum-likelihood fits of all models
    forecasts    main rolling-window out-of-sample forecasts
    robustness   alternative windows, refit frequency and pre-1996 history
    montecarlo   simulation experiment with calibrated DGPs
"""
from __future__ import annotations

import json
import tomllib
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import jarque_bera, kurtosis, skew
from statsmodels.stats.diagnostic import acorr_ljungbox

from .data import load_returns, sha256
from .forecasting import run_forecasts
from .models import MODELS
from .simulation import run_montecarlo

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path | None = None) -> dict:
    path = Path(path) if path else ROOT / "config.toml"
    with open(path, "rb") as f:
        return tomllib.load(f)


def results_dir() -> Path:
    p = ROOT / "results"
    p.mkdir(exist_ok=True)
    return p


def returns(cfg, start=None):
    c = cfg["data"]
    return load_returns(ROOT / c["raw_csv"], start or c["sample_start"], c["sample_end"],
                        expected_sha256=c["sha256"])


# ---------------------------------------------------------------------------
def stage_data(cfg):
    out = results_dir()
    d = returns(cfg)
    d.to_csv(out / "clean_returns.csv", index=False)
    full = returns(cfg, start=cfg["robustness"]["full_sample_start"])
    x = d.ret.to_numpy()
    lb = acorr_ljungbox(x, lags=[10], return_df=True)
    lb2 = acorr_ljungbox((x - x.mean()) ** 2, lags=[10], return_df=True)
    diag = {
        "raw_file_sha256": sha256(ROOT / cfg["data"]["raw_csv"]),
        "raw_rows": int(d.attrs["n_raw_rows"]), "raw_missing": int(d.attrs["n_missing"]),
        "start": str(d.date.min().date()), "end": str(d.date.max().date()), "n": int(len(d)),
        "mean": float(x.mean()), "sd": float(x.std(ddof=1)), "skewness": float(skew(x)),
        "excess_kurtosis": float(kurtosis(x)), "min": float(x.min()), "max": float(x.max()),
        "jb_p": float(jarque_bera(x).pvalue), "lb10_p": float(lb.lb_pvalue.iloc[0]),
        "lb10_sq_p": float(lb2.lb_pvalue.iloc[0]),
        "share_gaps_over_1day": float(np.mean(d.gap_days > 1)),
        "full_sample_start": str(full.date.min().date()), "full_n": int(len(full)),
        "full_skewness": float(skew(full.ret)), "full_excess_kurtosis": float(kurtosis(full.ret)),
        "full_max": float(full.ret.max()),
    }
    (out / "diagnostics.json").write_text(json.dumps(diag, indent=2))
    return diag


def qmle_standard_errors(model, theta, x, rel_step=1e-4):
    """Robust (sandwich) QMLE standard errors from numerical derivatives.

    H = Hessian of the average log-likelihood, J = outer product of the
    per-observation scores; Var(theta) = H^-1 J H^-1 / n.
    """
    theta = np.asarray(theta, float)
    k, n = len(theta), len(x)
    steps = rel_step * np.maximum(1.0, np.abs(theta))

    def scores(th):
        S = np.empty((n, k))
        for i in range(k):
            e = np.zeros(k); e[i] = steps[i]
            S[:, i] = (model.loglik_obs(th + e, x) - model.loglik_obs(th - e, x)) / (2 * steps[i])
        return S

    S = scores(theta)
    J = S.T @ S / n
    H = np.empty((k, k))
    for i in range(k):
        e = np.zeros(k); e[i] = steps[i]
        H[:, i] = (scores(theta + e).mean(0) - scores(theta - e).mean(0)) / (2 * steps[i])
    H = 0.5 * (H + H.T)
    Hinv = np.linalg.pinv(H)
    V = Hinv @ J @ Hinv / n
    return np.sqrt(np.clip(np.diag(V), 0, None))


def stage_insample(cfg):
    out = results_dir()
    x = returns(cfg).ret.to_numpy()
    rows = []
    for name in cfg["forecast"]["models"]:
        f = MODELS[name].fit(x)
        se = qmle_standard_errors(MODELS[name], f.theta, x)
        rows.append({"model": name, "k": f.k, "loglik": -f.nll, "aic": f.aic, "bic": f.bic,
                     "converged": f.success,
                     "params": json.dumps(dict(zip(MODELS[name].params, map(float, f.theta)))),
                     "se": json.dumps(dict(zip(MODELS[name].params, map(float, se))))})
    df = pd.DataFrame(rows)
    df.to_csv(out / "insample_fits.csv", index=False)
    return df


def _oos_first_date(cfg):
    d = returns(cfg)
    return d.date.iloc[cfg["forecast"]["window"]]


def _forecast_variant(cfg, start, window, refit_every, models, tag, force=False):
    """Forecasts for one design. Models already stored in results/ are reused
    unless force=True, so adding a model to config.toml only computes that model."""
    out = results_dir()
    f_fc, f_par = out / f"forecasts_{tag}.csv.gz", out / f"parameters_{tag}.csv"
    old_fc = old_par = None
    if f_fc.exists() and not force:
        old_fc = pd.read_csv(f_fc, parse_dates=["date"])
        old_par = pd.read_csv(f_par, parse_dates=["refit_date"])
        old_fc, old_par = old_fc[old_fc.model.isin(models)], old_par[old_par.model.isin(models)]
    todo = [m for m in models if old_fc is None or m not in set(old_fc.model)]
    if todo:
        d = returns(cfg, start=start)
        first = int(np.searchsorted(d.date.to_numpy(), np.datetime64(_oos_first_date(cfg))))
        fc, pars = run_forecasts(d.ret.to_numpy(), d.date, todo, first, window, refit_every,
                                 tuple(cfg["forecast"]["alphas"]))
        fc.insert(0, "variant", tag)
        pars.insert(0, "variant", tag)
        if old_fc is not None:
            fc, pars = pd.concat([old_fc, fc], ignore_index=True), pd.concat([old_par, pars], ignore_index=True)
        fc.to_csv(f_fc, index=False)
        pars.to_csv(f_par, index=False)
    print(f"[{tag}] computed: {todo or 'nothing (cached)'}", flush=True)


def stage_forecasts(cfg, force=False):
    f = cfg["forecast"]
    _forecast_variant(cfg, None, f["window"], f["refit_every"], f["models"], "main", force)


def stage_robustness(cfg, force=False):
    f, r = cfg["forecast"], cfg["robustness"]
    variants = [
        (None, r["alt_window"], f["refit_every"], f"window{r['alt_window']}"),
        (None, f["window"], r["alt_refit_every"], f"refit{r['alt_refit_every']}"),
        (r["full_sample_start"], None, 63, "expanding1993"),
    ]
    for s, w, k, tag in variants:
        _forecast_variant(cfg, s, w, k, f["models"], tag, force)


def stage_montecarlo(cfg):
    out = results_dir()
    mc = cfg["montecarlo"]
    fits = pd.read_csv(out / "insample_fits.csv").set_index("model")
    dgp_params = {m: list(json.loads(fits.loc[m, "params"]).values()) for m in mc["dgps"]}
    (out / "montecarlo_dgp_parameters.json").write_text(json.dumps(dgp_params, indent=2))
    scores, recovery = run_montecarlo(dgp_params, cfg["forecast"]["models"], mc["replications"],
                                      mc["n_train"], mc["n_test"], mc["seed"])
    scores.to_csv(out / "montecarlo_scores.csv", index=False)
    recovery.to_csv(out / "montecarlo_recovery.csv", index=False)
    return scores, recovery


STAGES = {"data": stage_data, "insample": stage_insample, "forecasts": stage_forecasts,
          "robustness": stage_robustness, "montecarlo": stage_montecarlo}
