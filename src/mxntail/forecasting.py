"""Rolling-window, one-step-ahead density and tail-risk forecasts."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .models import MODELS

# Every `full_restart_every`-th re-estimation uses all default starting values;
# the others start from the previous estimate plus the first default start.
FULL_RESTART_EVERY = 12


def forecast_model(name: str, x: np.ndarray, dates: pd.Series, first: int,
                   window: int | None, refit_every: int, alphas=(0.95, 0.99)):
    """Out-of-sample forecasts for returns x[first:], one model.

    window=None gives an expanding window starting at observation 0.
    Returns (daily forecast table, parameter table).
    """
    model = MODELS[name]
    n = len(x)
    rows, params = [], []
    warm = None
    taus = list(range(first, n, refit_every))
    for i, tau in enumerate(taus):
        if i % 50 == 0:
            print(f"  {name}: re-estimation {i + 1} of {len(taus)}", flush=True)
        start = 0 if window is None else tau - window
        hist = x[start:tau]
        block = x[tau:min(tau + refit_every, n)]
        full = (i % FULL_RESTART_EVERY == 0)
        fit = model.fit(hist, warm=warm, max_starts=None if full else 1)
        warm = fit.theta
        d = model.predictive(fit.theta, hist, block)
        out = {"date": dates.iloc[tau:tau + len(block)].to_numpy(), "y": block,
               "nls": -d.logpdf(block), "pit": d.cdf(block)}
        for a in alphas:
            q = d.quantile(a)
            tag = f"{round(100 * a)}"
            out[f"VaR{tag}"] = q
            out[f"ES{tag}"] = d.es(a, q)
        if model.conditional:
            out["cond_sd"] = d.center_scale()[1]
        rows.append(pd.DataFrame(out))
        params.append({"model": name, "refit_date": dates.iloc[tau], "n_train": len(hist),
                       "nll": fit.nll, "converged": fit.success,
                       **dict(zip(model.params, fit.theta))})
    df = pd.concat(rows, ignore_index=True)
    df.insert(0, "model", name)
    return df, pd.DataFrame(params)


def run_forecasts(x, dates, models, first, window, refit_every, alphas, workers=2):
    """Run `forecast_model` for several models in parallel processes."""
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {m: ex.submit(forecast_model, m, x, dates, first, window, refit_every, alphas)
                for m in models}
        results = {m: f.result() for m, f in futs.items()}
    fc = pd.concat([results[m][0] for m in models], ignore_index=True)
    pars = pd.concat([results[m][1] for m in models], ignore_index=True)
    return fc, pars
