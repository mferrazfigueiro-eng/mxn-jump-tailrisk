# Jumps or Asymmetric Tails? Lévy Jump-Diffusion Models and the Tail Risk of the Mexican Peso

Replication package for the article by **Michele Ferraz Figueiró** (Instituto Tecnológico y de Estudios Superiores de Monterrey),
submitted to *Applied Stochastic Models in Business and Industry*.

[![DOI](https://zenodo.org/badge/1463251227.svg)](https://doi.org/10.5281/zenodo.24353202)

Repository: <https://github.com/mferrazfigueiro-eng/mxn-jump-tailrisk> · Archived version: <https://doi.org/10.5281/zenodo.24353202>

One command reproduces every number, table and figure in the paper from the raw FRED file:

```bash
pip install -r requirements.txt
python scripts/run_all.py          # about 1.5 hours on a 2-core laptop
cd paper && latexmk -pdf main.tex  # compiles the manuscript from the generated outputs
```

The manuscript never contains typed-in results. `scripts/run_all.py` writes the tables
(`paper/tables/*.tex`), figures (`paper/figures/*.pdf`) and every number quoted in the
text (`paper/numbers.tex`, as LaTeX macros), and `paper/main.tex` only reads those files.

## What the study does

Daily USD/MXN log changes (January 1996 – October 2026) are forecast one day ahead with
eight models estimated by maximum likelihood:

| Independent increments (Lévy) | GARCH(1,1) volatility |
|---|---|
| Gaussian | GARCH-t |
| Student-t | GARCH-skew-t (Hansen, 1994) |
| Merton jump diffusion | GARCH-Merton (normal jumps) |
| Kou double-exponential jumps | GARCH-Kou (double-exponential jumps) |

Forecasts use a rolling 1,500-day window re-estimated every 21 days, and are evaluated with
log scores, the FZ0 joint VaR–ES loss, Kupiec and Christoffersen coverage tests,
Du–Escanciano ES backtests, Diebold–Mariano tests and the Model Confidence Set. A calibrated
Monte Carlo study and three robustness designs complete the analysis.

## Repository structure

```
config.toml              every setting used in the paper (sample, window, seeds, ...)
data/raw/DEXMXUS.csv     unmodified FRED download; checksum verified on every run
data/README.md           data source, download date, SHA-256 and terms of use
src/mxntail/
    data.py              loading and cleaning, returns in percent
    distributions.py     predictive densities, CDFs, VaR and ES (vectorised)
    models.py            the eight models: likelihoods, estimation, forecasting, simulation
    forecasting.py       rolling-window out-of-sample forecasts
    evaluation.py        losses, VaR/ES backtests, Diebold-Mariano, Model Confidence Set
    simulation.py        Monte Carlo experiment
    pipeline.py          computation stages (data, insample, forecasts, robustness, montecarlo)
    reporting.py         tables, figures and numbers.tex for the manuscript
scripts/run_all.py       runs all stages in order, with caching
notebooks/               one short notebook per part of the paper (see below)
tests/                   numerical unit tests (pytest)
results/                 generated outputs (CSV); safe to delete and regenerate
paper/                   main.tex, references.bib and generated tables/figures
```

## Where each result comes from

| Article | Notebook | Code | Output in `results/` |
|---|---|---|---|
| Table 1, Figure 1 | `01_data_and_diagnostics` | `pipeline.stage_data` | `diagnostics.json` |
| Table 2 | `02_insample_estimation` | `pipeline.stage_insample` | `insample_fits.csv` |
| Tables 3–4, Figures 2–5 | `03_out_of_sample_forecasts` | `pipeline.stage_forecasts`, `evaluation` | `forecasts_main.csv.gz`, `summary_main.csv`, `dm_main.csv`, `mcs_main.csv` |
| Tables 5–6 | `04_events_and_application` | `reporting.table_events`, `table_application` | `events_main.csv`, `application_main.csv` |
| Table 7 | `05_robustness` | `pipeline.stage_robustness` | `forecasts_window1000.csv.gz`, `forecasts_refit5.csv.gz`, `forecasts_expanding1993.csv.gz`, `robustness_summary.csv` |
| Table 8 | `06_monte_carlo` | `pipeline.stage_montecarlo` | `montecarlo_scores.csv`, `montecarlo_recovery.csv`, `montecarlo_summary.csv` |
| Check of all of the above | `00_verify_replication` | `scripts/verify_replication.py` | `verification_report.csv` |
| Manuscript PDF | `07_build_article` | `reporting.build_all` | `paper/` |

The notebooks are thin: they call the same functions as `run_all.py`, reuse cached
results when present, and display the tables. Run `make notebooks` to execute all of them.

## Verifying the results

`notebooks/00_verify_replication.ipynb` (or `python scripts/verify_replication.py`) deletes the
generated outputs, reruns the pipeline and compares every number quoted in the text, every
table and every result file with the copy shipped in the package. Set `FULL_CHECK = True` to
recompute everything from the raw data (about 1.5–2 hours); the quick check only rebuilds the
tables, figures and numbers from the saved results. A report is written to
`verification_report.csv`.

## Reproducibility notes

* **Data.** The exact FRED snapshot is included. FRED revises and extends series, so a fresh
  download will not reproduce the paper exactly; the code refuses to run if the checksum differs.
* **Randomness.** All random numbers come from seeds in `config.toml` (Monte Carlo draws and the
  MCS bootstrap). Maximum-likelihood fits use deterministic starting values.
* **Caching.** Stages already computed are reused. Adding a model to `config.toml` computes only
  that model. Use `python scripts/run_all.py --force` to recompute everything.
* **Numerical precision.** Results were produced with the package versions in `requirements.txt`
  (Python 3.13, Linux). Other BLAS libraries or platforms can change optimiser paths in the
  last digits; conclusions are unaffected.
* **Tests.** `python -m pytest -q` checks that every density integrates to one, that CDFs,
  quantiles and Expected Shortfall agree with numerical integration and closed forms, that
  estimators recover simulated parameters, and that GARCH forecasts use no future data.

## Citation

See `CITATION.cff`. Please cite the article and this package (DOI 10.5281/zenodo.24353202).

## License

Code: MIT (see `LICENSE`). Data: FRED series DEXMXUS, Board of Governors of the Federal
Reserve System; see `data/README.md`.
