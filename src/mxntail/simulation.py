"""Monte Carlo experiment with empirically calibrated data-generating processes.

Each replication simulates n_train + n_test returns from a DGP whose
parameters equal the full-sample USD/MXN estimates of that model, fits every
candidate model on the first n_train observations and scores one-step-ahead
forecasts on the last n_test observations (GARCH variances are filtered with
realised past returns, parameters are fixed).
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from .models import MODELS


def _one_replication(args):
    dgp, theta, rep, seed, n_train, n_test, models = args
    if rep % 50 == 0:
        print(f"  Monte Carlo: DGP {dgp}, replication {rep + 1}", flush=True)
    rng = np.random.default_rng([seed, rep, list(MODELS).index(dgp)])
    x = MODELS[dgp].simulate(np.asarray(theta), n_train + n_test, rng)
    tr, te = x[:n_train], x[n_train:]
    rows, recov = [], None
    for m in models:
        mod = MODELS[m]
        fit = mod.fit(tr, max_starts=2)
        d = mod.predictive(fit.theta, tr, te)
        q95, q99 = d.quantile(0.95), d.quantile(0.99)
        rows.append({"dgp": dgp, "rep": rep, "model": m, "nls": float(np.mean(-d.logpdf(te))),
                     "exc95": float(np.mean(te > q95)), "exc99": float(np.mean(te > q99)),
                     "converged": fit.success, "aic": fit.aic, "bic": fit.bic})
        if m == dgp and dgp in ("Merton", "GARCH-Merton"):
            recov = {"dgp": dgp, "rep": rep, **{f"hat_{k}": v for k, v in zip(mod.params, fit.theta)},
                     **{f"true_{k}": v for k, v in zip(mod.params, theta)}}
    return rows, recov


def run_montecarlo(dgp_params: dict, models, replications, n_train, n_test, seed, workers=2):
    tasks = [(dgp, th, r, seed, n_train, n_test, models)
             for dgp, th in dgp_params.items() for r in range(replications)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_one_replication, tasks, chunksize=4))
    scores = pd.DataFrame([r for rows, _ in out for r in rows])
    recovery = pd.DataFrame([rc for _, rc in out if rc is not None])
    return scores, recovery
