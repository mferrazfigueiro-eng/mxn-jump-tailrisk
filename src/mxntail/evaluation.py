"""Forecast evaluation: scoring rules, VaR/ES backtests and model comparison."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from arch.bootstrap import MCS
from scipy.stats import chi2, norm


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
def quantile_loss(y, var, alpha):
    """Tick loss for the upper-tail alpha-quantile."""
    return (alpha - (y < var).astype(float)) * (y - var)


def fz0_loss(y, var, es, alpha):
    """FZ0 joint VaR-ES loss (Patton, Ziegel and Chen, 2019), upper tail.

    The upper tail of y is the lower tail of x = -y at level a = 1 - alpha,
    with v = -VaR and e = -ES (<0). Lower values are better.
    """
    a = 1.0 - alpha
    x, v, e = -np.asarray(y), -np.asarray(var), -np.asarray(es)
    hit = (x <= v).astype(float)
    return -hit * (v - x) / (a * e) + v / e + np.log(-e) - 1.0


# ---------------------------------------------------------------------------
# VaR backtests
# ---------------------------------------------------------------------------
def kupiec_pof(hits, p):
    hits = np.asarray(hits, int)
    n, k = len(hits), int(hits.sum())
    ph = k / n
    ll0 = k * np.log(p) + (n - k) * np.log(1 - p)
    ll1 = (k * np.log(ph) if k else 0.0) + ((n - k) * np.log(1 - ph) if k < n else 0.0)
    lr = 2 * (ll1 - ll0)
    return lr, float(chi2.sf(lr, 1))


def christoffersen(hits, p):
    """Independence and conditional-coverage likelihood-ratio tests."""
    h = np.asarray(hits, int)
    a, b = h[:-1], h[1:]
    n00 = np.sum((a == 0) & (b == 0)); n01 = np.sum((a == 0) & (b == 1))
    n10 = np.sum((a == 1) & (b == 0)); n11 = np.sum((a == 1) & (b == 1))
    pi0 = n01 / max(n00 + n01, 1)
    pi1 = n11 / max(n10 + n11, 1)
    pi = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)

    def ll(p_, k, m):
        out = 0.0
        if k:
            out += k * np.log(p_)
        if m:
            out += m * np.log(1 - p_)
        return out
    lr_ind = 2 * (ll(pi0, n01, n00) + ll(pi1, n11, n10) - ll(pi, n01 + n11, n00 + n10))
    lr_uc, _ = kupiec_pof(h, p)
    lr_cc = lr_uc + lr_ind
    return float(chi2.sf(lr_ind, 1)), float(chi2.sf(lr_cc, 2))


# ---------------------------------------------------------------------------
# Expected Shortfall backtests (Du and Escanciano, 2017)
# ---------------------------------------------------------------------------
def du_escanciano(pit, alpha, lags=5):
    """Unconditional (U_ES) and conditional (C_ES) ES backtests, upper tail.

    u_t = 1 - F_t(y_t) is the upper-tail PIT; with a = 1 - alpha the cumulative
    violation H_t = (a - u_t)/a * 1{u_t <= a} has mean a/2 and variance
    a(1/3 - a/4) under correct specification and is serially uncorrelated.
    """
    a = 1.0 - alpha
    u = 1.0 - np.asarray(pit)
    H = (a - u) / a * (u <= a)
    n = len(H)
    stat_u = np.sqrt(n) * (H.mean() - a / 2) / np.sqrt(a * (1 / 3 - a / 4))
    p_u = float(2 * norm.sf(abs(stat_u)))
    hc = H - a / 2
    g0 = np.mean(hc * hc)
    rho = np.array([np.mean(hc[j:] * hc[:-j]) / g0 for j in range(1, lags + 1)])
    stat_c = n * np.sum(rho ** 2)
    p_c = float(chi2.sf(stat_c, lags))
    return float(H.mean()), float(stat_u), p_u, p_c


# ---------------------------------------------------------------------------
# Comparisons
# ---------------------------------------------------------------------------
def dm_hac(diff, lags=10):
    """Diebold-Mariano test: mean of a loss differential with Newey-West SE."""
    d = np.asarray(diff, float)
    fit = sm.OLS(d, np.ones((len(d), 1))).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
    return float(fit.params[0]), float(fit.bse[0]), float(fit.tvalues[0]), float(fit.pvalues[0])


def model_confidence_set(losses: pd.DataFrame, size=0.10, reps=2000, block=10, seed=0):
    """Hansen, Lunde and Nason (2011) MCS on a (days x models) loss table."""
    mcs = MCS(losses.to_numpy(), size=size, reps=reps, block_size=block,
              method="R", bootstrap="stationary", seed=seed)
    mcs.compute()
    pv = mcs.pvalues.copy()
    pv.index = [losses.columns[i] for i in pv.index]
    pv = pv.reindex(losses.columns)
    return pv["Pvalue"]


def add_losses(fc: pd.DataFrame, alphas) -> pd.DataFrame:
    fc = fc.copy()
    for a in alphas:
        t = f"{round(100 * a)}"
        fc[f"hit{t}"] = (fc["y"] > fc[f"VaR{t}"]).astype(int)
        fc[f"QL{t}"] = quantile_loss(fc["y"].to_numpy(), fc[f"VaR{t}"].to_numpy(), a)
        fc[f"FZ{t}"] = fz0_loss(fc["y"].to_numpy(), fc[f"VaR{t}"].to_numpy(),
                                fc[f"ES{t}"].to_numpy(), a)
    return fc


def summary_table(fc: pd.DataFrame, models, alphas) -> pd.DataFrame:
    rows = []
    for m in models:
        g = fc[fc.model == m]
        r = {"model": m, "n": len(g), "NLS": g.nls.mean()}
        for a in alphas:
            t = f"{round(100 * a)}"
            p = 1 - a
            hits = g[f"hit{t}"].to_numpy()
            _, p_uc = kupiec_pof(hits, p)
            p_ind, p_cc = christoffersen(hits, p)
            _, _, p_ues, p_ces = du_escanciano(g.pit.to_numpy(), a)
            r.update({f"exc{t}": int(hits.sum()), f"rate{t}": hits.mean(),
                      f"p_uc{t}": p_uc, f"p_ind{t}": p_ind, f"p_cc{t}": p_cc,
                      f"p_ues{t}": p_ues, f"p_ces{t}": p_ces,
                      f"QL{t}": g[f"QL{t}"].mean(), f"FZ{t}": g[f"FZ{t}"].mean()})
        rows.append(r)
    return pd.DataFrame(rows)


def comparison_table(fc: pd.DataFrame, models, benchmark, losses, lags=10,
                     mcs_size=0.10, mcs_reps=2000, mcs_block=10, seed=0):
    """DM-HAC versus `benchmark` and MCS p-values for each loss."""
    rows = []
    mcs_p = {}
    for loss in losses:
        wide = fc.pivot(index="date", columns="model", values=loss)[models].dropna()
        mcs_p[loss] = model_confidence_set(wide, mcs_size, mcs_reps, mcs_block, seed)
        for m in models:
            if m == benchmark:
                continue
            mean, se, t, p = dm_hac(wide[m] - wide[benchmark], lags)
            rows.append({"loss": loss, "model": m, "benchmark": benchmark,
                         "mean_diff": mean, "se": se, "t": t, "p": p})
    dm = pd.DataFrame(rows)
    mcs = pd.DataFrame(mcs_p)
    return dm, mcs
