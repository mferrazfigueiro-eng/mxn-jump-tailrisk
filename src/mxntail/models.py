"""Eight candidate models for daily USD/MXN log changes, estimated by maximum likelihood.

Unconditional (i.i.d. increments): Gaussian, Student-t, Merton, Kou.
Conditional (GARCH(1,1) volatility): GARCH-t, GARCH-skew-t, GARCH-Merton, GARCH-Kou.

For the GARCH jump models the conditional variance recursion is driven by
the total innovation eps_t = r_t - E[r_t], so a jump raises subsequent
diffusive volatility (constant-intensity ARCH-jump specification in the
spirit of Jorion, 1988):

    r_t = mu + sqrt(h_t) z_t + J_t,      h_t = omega + alpha eps_{t-1}^2 + beta h_{t-1}.

Each model exposes:
    nll(theta, x)                        negative log-likelihood
    fit(x, warm=None)                    MLE with deterministic multistart
    predictive(theta, hist, future)      one-step-ahead predictive distributions
                                         for every observation of `future`, using
                                         only data observed before each day
    simulate(theta, n, rng)              draw a sample path (Monte Carlo)
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.optimize import minimize

from .distributions import HansenSkewT, KouBernoulli, MertonMixture, Normal, StudentT

BIG = 1e12


@njit(cache=True)
def garch_filter(eps, omega, alpha, beta, h0):
    """h_t for t = 0..n-1, where h_t uses eps up to t-1 only (no look-ahead)."""
    n = eps.shape[0]
    h = np.empty(n)
    h[0] = h0
    for t in range(1, n):
        h[t] = omega + alpha * eps[t - 1] * eps[t - 1] + beta * h[t - 1]
        if h[t] < 1e-10:
            h[t] = 1e-10
    return h


def _ones(n):
    return 1.0 if n is None else np.ones(n)


@dataclass
class Fit:
    model: str
    theta: np.ndarray
    nll: float
    success: bool
    n: int

    @property
    def k(self) -> int:
        return len(self.theta)

    @property
    def aic(self) -> float:
        return 2 * self.nll + 2 * self.k

    @property
    def bic(self) -> float:
        return 2 * self.nll + np.log(self.n) * self.k


class Model:
    name = ""
    params: tuple[str, ...] = ()
    conditional = False

    def bounds(self, x):
        raise NotImplementedError

    def starts(self, x):
        raise NotImplementedError

    def dist(self, theta, h=None, n=None):
        """Predictive distribution; `h` = conditional variances, `n` = days (i.i.d. models)."""
        raise NotImplementedError

    def mean_shift(self, theta):
        """E[r_t] - mu (jump mean contribution), used for GARCH innovations."""
        return 0.0

    # -- likelihood ---------------------------------------------------------
    def _h(self, theta, x, h0=None):
        mu, omega, alpha, beta = theta[:4]
        eps = x - (mu + self.mean_shift(theta))
        if h0 is None:
            h0 = float(np.var(x))
        return garch_filter(eps, omega, alpha, beta, h0)

    def loglik_obs(self, theta, x):
        """Per-observation log-likelihood contributions."""
        theta = np.asarray(theta, float)
        h = self._h(theta, x) if self.conditional else None
        return self.dist(theta, h, n=None if self.conditional else len(x)).logpdf(x)

    def nll(self, theta, x):
        theta = np.asarray(theta, float)
        if self.conditional and theta[2] + theta[3] >= 0.9999:
            return BIG
        h = self._h(theta, x) if self.conditional else None
        ll = self.dist(theta, h).logpdf(x)
        val = -float(np.sum(ll))
        return val if np.isfinite(val) else BIG

    def fit(self, x, warm=None, max_starts=None, maxiter=500) -> Fit:
        """MLE by L-BFGS-B from deterministic starting values.

        `warm` (e.g. the previous re-estimation) is tried first; `max_starts`
        limits the number of default starting values that are also tried.
        """
        x = np.asarray(x, float)
        bnds = self.bounds(x)
        lo = np.array([b[0] for b in bnds])
        hi = np.array([b[1] for b in bnds])
        cands = [np.clip(np.asarray(s, float), lo, hi) for s in self.starts(x)]
        if max_starts is not None:
            cands = cands[:max_starts]
        if warm is not None:
            cands.insert(0, np.clip(np.asarray(warm, float), lo, hi))
        best = None
        for s in cands:
            r = minimize(self.nll, s, args=(x,), method="L-BFGS-B", bounds=bnds,
                         options={"maxiter": maxiter, "ftol": 1e-10, "gtol": 1e-6})
            if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                best = r
        return Fit(self.name, best.x, float(best.fun), bool(best.success), len(x))

    # -- forecasting --------------------------------------------------------
    def predictive(self, theta, hist, future):
        """Predictive distributions for each element of `future`.

        Conditional models filter the variance through `hist` (the estimation
        window, starting from its sample variance) and then through `future`,
        so the forecast for day t uses realised returns up to t-1 only.
        """
        theta = np.asarray(theta, float)
        future = np.asarray(future, float)
        if not self.conditional:
            return self.dist(theta, None, n=len(future))
        z = np.concatenate([hist, future])
        h = self._h(theta, z, h0=float(np.var(hist)))
        return self.dist(theta, h[len(hist):])


# ---------------------------------------------------------------------------
# Unconditional models
# ---------------------------------------------------------------------------
class Gaussian(Model):
    name = "Gaussian"
    params = ("mu", "sigma")

    def bounds(self, x):
        return [(-5, 5), (1e-4, 20)]

    def starts(self, x):
        return [[np.mean(x), np.std(x)]]

    def dist(self, theta, h=None, n=None):
        o = _ones(n)
        return Normal(theta[0] * o, theta[1] * o)

    def fit(self, x, warm=None, max_starts=None, maxiter=500):
        th = np.array([np.mean(x), np.std(x)])
        return Fit(self.name, th, self.nll(th, x), True, len(x))

    def simulate(self, theta, n, rng):
        return rng.normal(theta[0], theta[1], n)


class StudentTModel(Model):
    name = "Student-t"
    params = ("nu", "mu", "scale")

    def bounds(self, x):
        return [(2.05, 200), (-5, 5), (1e-4, 20)]

    def starts(self, x):
        med, s = np.median(x), np.std(x)
        return [[5.0, med, 0.6 * s], [3.0, med, 0.4 * s]]

    def dist(self, theta, h=None, n=None):
        o = _ones(n)
        return StudentT(theta[0] * o, theta[1] * o, theta[2] * o)


class Merton(Model):
    name = "Merton"
    params = ("b", "sigma", "lambda", "muJ", "deltaJ")

    def bounds(self, x):
        return [(-5, 5), (1e-3, 20), (1e-4, 2.0), (-10, 10), (1e-3, 20)]

    def starts(self, x):
        m, s = np.mean(x), np.std(x)
        return [[m, 0.7 * s, 0.05, 0.5 * s, 2.0 * s],
                [m, 0.8 * s, 0.02, 1.0 * s, 3.0 * s],
                [m, 0.6 * s, 0.20, 0.0, 1.5 * s],
                [m, 0.9 * s, 0.01, -0.5 * s, 3.0 * s]]

    def dist(self, theta, h=None, n=None):
        o = _ones(n)
        return MertonMixture(*(v * o for v in theta[:5]))

    def simulate(self, theta, n, rng):
        b, s, lam, mj, dj = theta
        k = rng.poisson(lam, n)
        return b + k * mj + rng.normal(size=n) * np.sqrt(s * s + k * dj * dj)


class Kou(Model):
    name = "Kou"
    params = ("b", "sigma", "q", "p_up", "eta_up", "eta_dn")

    def bounds(self, x):
        return [(-5, 5), (1e-3, 20), (1e-4, 0.5), (1e-3, 1 - 1e-3), (0.05, 50), (0.05, 50)]

    def starts(self, x):
        m, s = np.mean(x), np.std(x)
        return [[m, 0.7 * s, 0.10, 0.6, 1 / s, 1 / s],
                [m, 0.8 * s, 0.03, 0.7, 0.5 / s, 1 / s],
                [m, 0.6 * s, 0.25, 0.5, 1.5 / s, 1.5 / s]]

    def dist(self, theta, h=None, n=None):
        o = _ones(n)
        return KouBernoulli(*(v * o for v in theta[:6]))

    def simulate(self, theta, n, rng):
        b, s, q, p, eu, ed = theta
        jump = rng.random(n) < q
        up = rng.random(n) < p
        size = np.where(up, rng.exponential(1 / eu, n), -rng.exponential(1 / ed, n))
        return b + s * rng.normal(size=n) + jump * size


# ---------------------------------------------------------------------------
# GARCH(1,1) models
# ---------------------------------------------------------------------------
def _garch_bounds(x):
    v = float(np.var(x))
    return [(-5, 5), (1e-7, 5 * v), (1e-4, 0.5), (1e-4, 0.9995)]


def _garch_starts(x):
    v = float(np.var(x))
    return [[np.mean(x), 0.03 * v, 0.06, 0.92], [np.mean(x), 0.10 * v, 0.12, 0.80]]


class GarchT(Model):
    name = "GARCH-t"
    params = ("mu", "omega", "alpha", "beta", "nu")
    conditional = True

    def bounds(self, x):
        return _garch_bounds(x) + [(2.05, 200)]

    def starts(self, x):
        return [s + [6.0] for s in _garch_starts(x)]

    def dist(self, theta, h=None, n=None):
        nu = theta[4]
        return StudentT(nu, theta[0], np.sqrt(h * (nu - 2) / nu))

    def simulate(self, theta, n, rng, burn=500):
        mu, om, a, b, nu = theta
        z = rng.standard_t(nu, n + burn) * np.sqrt((nu - 2) / nu)
        r = np.empty(n + burn)
        h = om / (1 - a - b)
        prev = 0.0
        for t in range(n + burn):
            if t:
                h = om + a * prev ** 2 + b * h
            prev = np.sqrt(h) * z[t]
            r[t] = mu + prev
        return r[burn:]


class GarchSkewT(Model):
    """GARCH(1,1) with Hansen (1994) skewed-t innovations: asymmetric tails without jumps."""
    name = "GARCH-skew-t"
    params = ("mu", "omega", "alpha", "beta", "nu", "lam")
    conditional = True

    def bounds(self, x):
        return _garch_bounds(x) + [(2.05, 200), (-0.95, 0.95)]

    def starts(self, x):
        return [s + [6.0, 0.05] for s in _garch_starts(x)]

    def dist(self, theta, h=None, n=None):
        return HansenSkewT(theta[4], theta[5], theta[0], np.sqrt(h))

    def simulate(self, theta, n, rng, burn=500):
        mu, om, a, b, nu, lam = theta
        d = HansenSkewT(nu, lam, 0.0, 1.0)
        # inverse-CDF sampling of standardised innovations
        u = rng.random(n + burn)
        grid = np.linspace(-30, 30, 60001)
        z = np.interp(u, d.cdf(grid), grid)
        r = np.empty(n + burn)
        h = om / (1 - a - b)
        prev = 0.0
        for t in range(n + burn):
            if t:
                h = om + a * prev ** 2 + b * h
            prev = np.sqrt(h) * z[t]
            r[t] = mu + prev
        return r[burn:]


class GarchMerton(Model):
    name = "GARCH-Merton"
    params = ("mu", "omega", "alpha", "beta", "lambda", "muJ", "deltaJ")
    conditional = True

    def bounds(self, x):
        return _garch_bounds(x) + [(1e-4, 2.0), (-10, 10), (1e-3, 20)]

    def starts(self, x):
        s = float(np.std(x))
        jumps = [[0.05, 0.5 * s, 2.0 * s], [0.02, 1.0 * s, 3.0 * s], [0.15, 0.0, 1.5 * s]]
        return [g + j for g in _garch_starts(x) for j in jumps]

    def mean_shift(self, theta):
        return theta[4] * theta[5]

    def dist(self, theta, h=None, n=None):
        return MertonMixture(theta[0], np.sqrt(h), theta[4], theta[5], theta[6])

    def simulate(self, theta, n, rng, burn=500):
        mu, om, a, b, lam, mj, dj = theta
        jv = lam * (mj ** 2 + dj ** 2)
        h = max((om + a * jv) / (1 - a - b), 1e-8)
        r = np.empty(n + burn)
        prev = 0.0
        for t in range(n + burn):
            if t:
                h = om + a * prev ** 2 + b * h
            k = rng.poisson(lam)
            jump = k * mj + np.sqrt(k) * dj * rng.normal() if k else 0.0
            prev = np.sqrt(h) * rng.normal() + jump - lam * mj
            r[t] = mu + lam * mj + prev
        return r[burn:]


class GarchKou(Model):
    name = "GARCH-Kou"
    params = ("mu", "omega", "alpha", "beta", "q", "p_up", "eta_up", "eta_dn")
    conditional = True

    def bounds(self, x):
        return _garch_bounds(x) + [(1e-4, 0.5), (1e-3, 1 - 1e-3), (0.05, 50), (0.05, 50)]

    def starts(self, x):
        s = float(np.std(x))
        jumps = [[0.05, 0.6, 1 / s, 1 / s], [0.02, 0.7, 0.5 / s, 1 / s], [0.15, 0.5, 1.5 / s, 1.5 / s]]
        return [g + j for g in _garch_starts(x) for j in jumps]

    def mean_shift(self, theta):
        q, p, eu, ed = theta[4:]
        return q * (p / eu - (1 - p) / ed)

    def dist(self, theta, h=None, n=None):
        return KouBernoulli(theta[0], np.sqrt(h), *theta[4:])


MODELS: dict[str, Model] = {m.name: m for m in [
    Gaussian(), StudentTModel(), Merton(), Kou(), GarchT(), GarchSkewT(), GarchMerton(), GarchKou()]}
