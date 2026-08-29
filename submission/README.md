# Content Performance Archetypes

## Project structure

- `work/notebooks/` — the five notebooks that make up the analysis, run in a fixed order (see below).
- `work/tests/` — a pytest suite that validates the CSVs the notebooks produce.
- `paper/archetype_paper.pdf` — the write-up.

Running the notebooks creates a `work/interim/` folder of intermediate CSVs. That folder is not checked into this repository, so it will not exist until you run the notebooks yourself.
## Requirements

- Python 3.12
- `duckdb`, `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `pytest`

```bash
pip install duckdb pandas numpy matplotlib scikit-learn pytest
```

If your machine's default `python`/`pytest` command resolves to a different, unconfigured Python install, use `py -3.12` (or the equivalent for whichever install has the packages above) in place of `python` in the commands below.

## HuggingFace access

Notebooks `01_rule_based_baseline.ipynb` and `04_data_leakage_audit.ipynb` read the source data directly from the HuggingFace dataset.

```python
con.execute("CREATE SECRET (TYPE huggingface, TOKEN 'hf_your_read_token')")
```

`hf_your_read_token` is a placeholder. Before running either notebook:

1. Visit the dataset's page on HuggingFace.
2. Generate a HuggingFace access token with read scope.
3. Replace `hf_your_read_token` in both notebooks with that token.

## Running the notebooks

The notebooks have a strict dependency order: each one reads CSVs that an earlier notebook writes to `work/interim/`.

1. **`01_rule_based_baseline.ipynb`** — pulls the raw data from HuggingFace, builds the rule-based archetype labels, and writes `page_level.csv`. This is the only notebook that touches the raw data source besides notebook 04's own diagnostic re-check.
2. **`02_kmeans_clustering.ipynb`** — reads `page_level.csv`, builds the client-grouped train/test split, fits the K-Means model, and writes `model_data.csv`, `train.csv`, `test.csv`, and `best_archetype_table.csv`.
3. **`03_model_validation.ipynb`**, **`04_data_leakage_audit.ipynb`**, **`05_split_strategy_comparison.ipynb`** — each reads the outputs of notebooks 01 and 02. They can run in any order relative to each other, but only after 01 and 02 have both completed.

Run 01, then 02, before any of the other three. `work/interim/` is created automatically on first write; no manual setup is needed.

## Running the tests

```bash
pytest work/tests/ -v
```
