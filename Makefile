# Convenience targets. Every target can also be run with the commands shown.
PY ?= python

all: results paper

results:            ## run every computation stage, then tables and figures (~1.5 h)
	$(PY) scripts/run_all.py

report:             ## rebuild tables, figures and numbers.tex from cached results
	$(PY) scripts/run_all.py --stages report

paper: report       ## compile the manuscript PDF
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

notebooks:          ## execute all notebooks in place (uses cached results)
	jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

test:               ## numerical unit tests
	$(PY) -m pytest -q

clean:              ## delete generated results (raw data are never touched)
	rm -rf results/* paper/tables/* paper/figures/* paper/numbers.tex

.PHONY: all results report paper notebooks test clean
