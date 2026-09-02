# photovoltaic_prediction

Supplementary code for the manuscript **"Design for low-carbon residential block considering BIPV self-consumption and lifecycle economic performance"** (under review).

Author: Hantao He

## What this project does
This project builds a machine-learning **surrogate model** of building energy and PV performance, then runs **multi-objective optimization** (NSGA-II) to co-optimize BIPV deployment and residential-block morphology for **self-consumption (SC)**, **net present value (NPV)** and **energy ratio (ER)** in a hot-summer climate (Shenzhen).

## Pipeline
| Step | Files | Description |
|---|---|---|
| 0. Parametric modeling + EnergyPlus simulation | *(not in repo: Rhino/Grasshopper + EnergyPlus, raw outputs excluded)* | Generate block variants and hourly building / PV performance |
| 1. Data preparation | `clean_csv.py` | Clean/reshape EnergyPlus hourly outputs into feature & target columns |
| 2. Surrogate training | `model_training.ipynb` | Leakage-free training (split by building, GroupKFold, OOF), GES model selection + weighted ensemble; saves models for the optimizer |
| 3. Multi-objective optimization | `multi-objective_optimization.ipynb` | NSGA-II (pymoo) over design variables; SVF modeled as a function of geometry; exports Pareto-optimal solutions |
| 4. Analysis & figures | `LHS visualization.py`, `plot_code/LHS.py` | Illustrations of the Latin-hypercube sampling used to generate design variants |

> Note: the trained `*.joblib` models and simulation/result CSV files are **not** included in this repository. Run the pipeline in order (or re-train in the notebook) before executing the optimization notebook.

## How to run
```bash
# 1) prepare data
python clean_csv.py
# 2) train surrogates (in Jupyter)
jupyter notebook model_training.ipynb
# 3) multi-objective optimization (in Jupyter)
jupyter notebook multi-objective_optimization.ipynb
```

## Requirements
Python 3.9+; `pandas`, `numpy`, `scikit-learn`, `xgboost`, `lightgbm`, `catboost`, `pymoo`, `matplotlib`, `seaborn`, `joblib`. (See the notebooks for exact versions used.)
