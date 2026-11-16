PY ?= .venv/bin/python

.PHONY: setup data test run figures notebook all clean

setup:            ## create the virtualenv and install dependencies
	python3 -m venv .venv
	.venv/bin/pip install -r requirements.txt

data:             ## download + checksum-verify the Dry Bean dataset
	PYTHONPATH=src $(PY) -c "from abtest.data import download; download()"

test:             ## unit tests for the statistics code
	$(PY) -m pytest -q

run:              ## full experiment: CV, hold-out, tests, tables, figures (about 10 min)
	$(PY) scripts/run_experiment.py

figures:          ## rebuild figures from the cached results (no retraining)
	PYTHONPATH=src $(PY) -c "import pickle; from abtest.plots import make_all_figures; make_all_figures(pickle.load(open('data/results_cache.pkl','rb')))"

notebook:         ## execute the analysis notebook in place
	$(PY) -m jupyter nbconvert --to notebook --execute --inplace notebooks/analysis.ipynb

all: test run notebook

clean:
	rm -rf results/*.csv results/*.json results/figures/*.png data/results_cache.pkl
