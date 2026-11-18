"""Numerical checks for densities, risk measures and estimators. Run: pytest -q"""
import numpy as np
import pytest
from scipy import integrate, stats

from mxntail.distributions import HansenSkewT, KouBernoulli, MertonMixture, Normal, StudentT
from mxntail.models import MODELS

DISTS = {
    "normal": Normal(np.array([0.1]), np.array([0.8])),
    "t": StudentT(np.array([4.0]), np.array([0.0]), np.array([0.5])),
    "merton": MertonMixture(np.array([0.0]), np.array([0.5]), np.array([0.1]),
                            np.array([0.4]), np.array([1.5])),
    "skewt": HansenSkewT(np.array([5.0]), np.array([0.3]), np.array([0.1]), np.array([0.7])),
    "kou": KouBernoulli(np.array([0.0]), np.array([0.5]), np.array([0.1]),
                        np.array([0.6]), np.array([1.2]), np.array([2.0])),
}


@pytest.mark.parametrize("name", DISTS)
def test_density_integrates_and_matches_cdf(name):
    d = DISTS[name]
    f = lambda x: float(np.exp(d.logpdf(np.array([x])))[0])
    total, _ = integrate.quad(f, -np.inf, np.inf, limit=400)
    assert abs(total - 1) < 1e-6
    for x in (-1.0, 0.3, 2.0):
        num, _ = integrate.quad(f, -np.inf, x, limit=400)
        assert abs(num - d.cdf(np.array([x]))[0]) < 1e-6


@pytest.mark.parametrize("name", DISTS)
@pytest.mark.parametrize("alpha", [0.95, 0.99])
def test_quantile_and_es(name, alpha):
    d = DISTS[name]
    q = d.quantile(alpha)
    assert abs(d.cdf(q)[0] - alpha) < 1e-7
    f = lambda x: x * float(np.exp(d.logpdf(np.array([x])))[0])
    tail, _ = integrate.quad(f, q[0], np.inf, limit=400)
    assert abs(d.es(alpha, q)[0] - tail / (1 - alpha)) < 1e-4


def test_normal_and_t_es_closed_form():
    a = 0.99
    n = Normal(np.array([0.0]), np.array([1.0]))
    assert abs(n.es(a)[0] - stats.norm.pdf(stats.norm.ppf(a)) / (1 - a)) < 1e-6
    v = 5.0
    t = StudentT(np.array([v]), np.array([0.0]), np.array([1.0]))
    qa = stats.t.ppf(a, v)
    es = stats.t.pdf(qa, v) * (v + qa ** 2) / (v - 1) / (1 - a)
    assert abs(t.es(a)[0] - es) < 1e-5


def test_kou_simulation_matches_cdf():
    m = MODELS["Kou"]
    th = np.array([0.0, 0.5, 0.1, 0.6, 1.2, 2.0])
    x = m.simulate(th, 400_000, np.random.default_rng(1))
    d = m.dist(th, n=1)
    for c in (-1.5, 0.0, 1.0, 3.0):
        assert abs(np.mean(x <= c) - d.cdf(np.array([c]))[0]) < 3e-3


@pytest.mark.parametrize("name,theta", [
    ("Merton", [0.0, 0.45, 0.08, 0.4, 1.4]),
    ("GARCH-t", [0.01, 0.01, 0.07, 0.90, 6.0]),
    ("GARCH-Merton", [0.0, 0.01, 0.06, 0.90, 0.05, 0.5, 1.2]),
])
def test_parameter_recovery(name, theta):
    m = MODELS[name]
    x = m.simulate(np.array(theta), 6000, np.random.default_rng(7))
    fit = m.fit(x)
    assert fit.success or np.isfinite(fit.nll)
    assert fit.nll <= m.nll(np.array(theta), x) + 1e-6  # MLE beats the truth in-sample
    if name == "GARCH-t":
        assert abs(fit.theta[2] + fit.theta[3] - 0.97) < 0.03


def test_no_lookahead_in_garch_forecasts():
    m = MODELS["GARCH-Merton"]
    th = np.array([0.0, 0.01, 0.06, 0.90, 0.05, 0.5, 1.2])
    rng = np.random.default_rng(3)
    hist, fut = rng.normal(size=500), rng.normal(size=20)
    d1 = m.predictive(th, hist, fut)
    fut2 = fut.copy()
    fut2[10:] += 50.0  # altering day >=10 must not change forecasts for days <=10
    d2 = m.predictive(th, hist, fut2)
    assert np.allclose(d1.s[:11], d2.s[:11])
    assert not np.allclose(d1.s[11:], d2.s[11:])


def test_skewt_standardised():
    d = HansenSkewT(np.array([6.0]), np.array([0.4]), np.array([0.0]), np.array([1.0]))
    f = lambda x, p: x ** p * float(np.exp(d.logpdf(np.array([x])))[0])
    m1, _ = integrate.quad(f, -np.inf, np.inf, args=(1,), limit=400)
    m2, _ = integrate.quad(f, -np.inf, np.inf, args=(2,), limit=400)
    assert abs(m1) < 1e-6 and abs(m2 - 1) < 1e-5
