Code and data to reproduce the figures and summary tables in *Solver Choice on a Learned Nonlinear Manifold: Error-Residual Tradeoffs in
Reduced-Order Modeling*.


## Requirements

- Python >= 3.11
- `numpy`, `scipy`, `torch`, `matplotlib` (declared in `pyproject.toml`)

Install the `mgp` package in editable mode from this directory:

```
pip install -e .
```

This is required for `export_solution_fields.py`, which imports `mgp`. The
figure and summary scripts use only the standard library plus `matplotlib`, so
they also run with `PYTHONPATH=src` if you prefer not to install.

## Layout

```
scripts/    publication scripts (figures, summaries, solution fields)
src/    model and benchmark source package
artifacts/  aggregated input data (study JSONs, summary CSV/JSON, one offline artifact)
```
