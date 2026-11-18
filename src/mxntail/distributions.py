"""One-step-ahead predictive distributions.

Every class stores parameter *arrays* (one entry per forecast day) so that
log scores, CDFs, quantiles (VaR) and Expected Shortfall can be evaluated for
thousands of days at once. All quantities refer to percentage log changes in
USD/MXN; the upper tail corresponds to peso depreciation.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln, log_ndtr, logsumexp, ndtr, stdtr

LOG2PI = np.log(2.0 * np.pi)
# Gauss-Legendre nodes for the Expected Shortfall integral.
_GL_T, _GL_W = np.polynomial.legendre.leggauss(400)
_GL_T = 0.5 * (_GL_T + 1.0)
_GL_W = 0.5 * _GL_W


def _norm_logpdf(x, m, s):
    z = (x - m) / s
    return -0.5 * z * z - np.log(s) - 0.5 * LOG2PI


class Predictive:
    """Base class. Subclasses implement logpdf, cdf and a rough scale for root finding."""

    def logpdf(self, y):
        raise NotImplementedError

    def cdf(self, x):
        raise NotImplementedError

    def center_scale(self):
        raise NotImplementedError

    def sf(self, x):
        return 1.0 - self.cdf(x)

    def quantile(self, alpha: float, tol: float = 1e-9) -> np.ndarray:
        """Vectorised bisection for F^{-1}(alpha), one value per forecast day."""
        c, s = self.center_scale()
        lo = c - 60.0 * s
        hi = c + 60.0 * s
        for _ in range(200):
            mid = 0.5 * (lo + hi)
            below = self.cdf(mid) < alpha
            lo = np.where(below, mid, lo)
            hi = np.where(below, hi, mid)
            if np.max(hi - lo) < tol:
                break
        return 0.5 * (lo + hi)

    def es(self, alpha: float, q: np.ndarray | None = None) -> np.ndarray:
        """Upper-tail Expected Shortfall: ES = q + (1/(1-alpha)) * int_q^inf S(x) dx.

        The integral uses the substitution x = q + c t/(1-t), t in [0,1), with
        c a scale of the distribution, and 400-point Gauss-Legendre quadrature.
        """
        if q is None:
            q = self.quantile(alpha)
        _, c = self.center_scale()
        t = _GL_T[:, None]
        x = q[None, :] + c[None, :] * t / (1.0 - t)
        jac = c[None, :] / (1.0 - t) ** 2
        surv = self._sf_grid(x)
        integral = np.sum(_GL_W[:, None] * surv * jac, axis=0)
        return q + integral / (1.0 - alpha)

    def _sf_grid(self, x):
        """Survival function on a (nodes x days) grid; broadcasting parameters over rows."""
        return 1.0 - self.cdf(x)


class Normal(Predictive):
    def __init__(self, m, s):
        self.m, self.s = np.broadcast_arrays(np.asarray(m, float), np.asarray(s, float))

    def logpdf(self, y):
        return _norm_logpdf(y, self.m, self.s)

    def cdf(self, x):
        return ndtr((x - self.m) / self.s)

    def _sf_grid(self, x):
        return ndtr(-(x - self.m) / self.s)

    def center_scale(self):
        return self.m, self.s


class StudentT(Predictive):
    """Location-scale Student-t with `df` degrees of freedom and scale `scale`."""

    def __init__(self, df, loc, scale):
        self.df, self.loc, self.scale = np.broadcast_arrays(
            np.asarray(df, float), np.asarray(loc, float), np.asarray(scale, float))

    def logpdf(self, y):
        v = self.df
        z = (y - self.loc) / self.scale
        return (gammaln(0.5 * (v + 1)) - gammaln(0.5 * v) - 0.5 * np.log(v * np.pi)
                - np.log(self.scale) - 0.5 * (v + 1) * np.log1p(z * z / v))

    def cdf(self, x):
        return stdtr(self.df, (x - self.loc) / self.scale)

    def _sf_grid(self, x):
        return stdtr(self.df, -(x - self.loc) / self.scale)

    def center_scale(self):
        sd = self.scale * np.sqrt(self.df / np.maximum(self.df - 2.0, 1e-3))
        return self.loc, sd


class HansenSkewT(Predictive):
    """Hansen (1994) skewed Student-t, standardised to mean 0 and variance 1.

    r = loc + scale * z, with z ~ skew-t(nu, lam); lam > 0 gives a heavier
    right (depreciation) tail.
    """

    def __init__(self, nu, lam, loc, scale):
        self.nu, self.lam, self.loc, self.scale = np.broadcast_arrays(
            *(np.asarray(a, float) for a in (nu, lam, loc, scale)))
        v = self.nu
        self.c = np.exp(gammaln(0.5 * (v + 1)) - gammaln(0.5 * v)) / np.sqrt(np.pi * (v - 2))
        self.a = 4 * self.lam * self.c * (v - 2) / (v - 1)
        self.b = np.sqrt(1 + 3 * self.lam ** 2 - self.a ** 2)

    def _z(self, x):
        return (x - self.loc) / self.scale

    def logpdf(self, y):
        z = self._z(y)
        v, lam = self.nu, self.lam
        side = np.where(z < -self.a / self.b, 1 - lam, 1 + lam)
        u = (self.b * z + self.a) / side
        return (np.log(self.b) + np.log(self.c) - 0.5 * (v + 1) * np.log1p(u * u / (v - 2))
                - np.log(self.scale))

    def cdf(self, x):
        z = self._z(x)
        v, lam = self.nu, self.lam
        k = np.sqrt(v / (v - 2))
        w = self.b * z + self.a
        left = (1 - lam) * stdtr(v, k * w / (1 - lam))
        right = 0.5 * (1 - lam) + (1 + lam) * (stdtr(v, k * w / (1 + lam)) - 0.5)
        return np.where(z < -self.a / self.b, left, right)

    def center_scale(self):
        return self.loc, self.scale


class MertonMixture(Predictive):
    """Poisson mixture of normals: sum_k Pois(k; lam) N(m + k muJ, s^2 + k dJ^2)."""

    def __init__(self, m, s, lam, muJ, dJ, kmax: int = 30):
        self.m, self.s, self.lam, self.muJ, self.dJ = np.broadcast_arrays(
            *(np.asarray(a, float) for a in (m, s, lam, muJ, dJ)))
        k = np.arange(kmax + 1, dtype=float)[:, None]
        self.k = k
        lam_ = np.maximum(self.lam, 1e-300)
        self.logw = k * np.log(lam_) - lam_ - gammaln(k + 1)
        self.means = self.m + k * self.muJ
        self.sds = np.sqrt(self.s ** 2 + k * self.dJ ** 2)

    def logpdf(self, y):
        return logsumexp(self.logw + _norm_logpdf(y, self.means, self.sds), axis=0)

    def cdf(self, x):
        return np.sum(np.exp(self.logw) * ndtr((x - self.means) / self.sds), axis=0)

    def _sf_grid(self, x):
        # x: (nodes, days) -> evaluate per mixture component
        w = np.exp(self.logw)[:, None, :]
        z = (x[None, :, :] - self.means[:, None, :]) / self.sds[:, None, :]
        return np.sum(w * ndtr(-z), axis=0)

    def center_scale(self):
        mean = self.m + self.lam * self.muJ
        var = self.s ** 2 + self.lam * (self.muJ ** 2 + self.dJ ** 2)
        return mean, np.sqrt(var)


def _emg_up_logpdf(x, m, s, eta):
    """log density of N(m, s^2) + Exp(rate=eta)."""
    return (np.log(eta) + eta * (m - x) + 0.5 * eta * eta * s * s
            + log_ndtr((x - m - eta * s * s) / s))


def _emg_up_cdf(x, m, s, eta):
    z = (x - m) / s
    tail = np.exp(-eta * (x - m) + 0.5 * eta * eta * s * s + log_ndtr(z - eta * s))
    return ndtr(z) - tail


class KouBernoulli(Predictive):
    """Normal diffusion plus at most one double-exponential jump per interval.

    With probability q a jump occurs; it is upward ~ Exp(eta_up) with
    probability p and downward ~ -Exp(eta_dn) otherwise (Kou, 2002). Allowing at
    most one jump per day (a Bernoulli approximation to the Poisson count)
    gives closed-form densities built from exponentially modified Gaussians.
    """

    def __init__(self, m, s, q, p, eta_up, eta_dn):
        self.m, self.s, self.q, self.p, self.eu, self.ed = np.broadcast_arrays(
            *(np.asarray(a, float) for a in (m, s, q, p, eta_up, eta_dn)))

    def logpdf(self, y):
        q = np.clip(self.q, 1e-12, 1 - 1e-12)
        p = np.clip(self.p, 1e-12, 1 - 1e-12)
        a = np.log1p(-q) + _norm_logpdf(y, self.m, self.s)
        b = np.log(q) + np.log(p) + _emg_up_logpdf(y, self.m, self.s, self.eu)
        # downward jump: mirror image of an upward EMG with mean -m evaluated at -y
        c = np.log(q) + np.log1p(-p) + _emg_up_logpdf(-y, -self.m, self.s, self.ed)
        return np.logaddexp(np.logaddexp(a, b), c)

    def cdf(self, x):
        n = ndtr((x - self.m) / self.s)
        up = _emg_up_cdf(x, self.m, self.s, self.eu)
        dn = 1.0 - _emg_up_cdf(-x, -self.m, self.s, self.ed)
        return (1 - self.q) * n + self.q * (self.p * up + (1 - self.p) * dn)

    def center_scale(self):
        jm = self.p / self.eu - (1 - self.p) / self.ed
        j2 = 2 * self.p / self.eu ** 2 + 2 * (1 - self.p) / self.ed ** 2
        mean = self.m + self.q * jm
        var = self.s ** 2 + self.q * j2 - (self.q * jm) ** 2
        return mean, np.sqrt(var)
