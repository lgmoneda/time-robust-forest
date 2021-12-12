# time-robust-forest

<div align="center">

[![Build status](https://github.com/lgmoneda/time-robust-forest/workflows/build/badge.svg?branch=main&event=push)](https://github.com/lgmoneda/time-robust-forest/actions?query=workflow%3Abuild)
[![Python Version](https://img.shields.io/pypi/pyversions/time-robust-forest.svg)](https://pypi.org/project/time-robust-forest/)
[![Dependencies Status](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen.svg)](https://github.com/lgmoneda/time-robust-forest/pulls?utf8=%E2%9C%93&q=is%3Apr%20author%3Aapp%2Fdependabot)

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Security: bandit](https://img.shields.io/badge/security-bandit-green.svg)](https://github.com/PyCQA/bandit)
[![Pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/lgmoneda/time-robust-forest/blob/main/.pre-commit-config.yaml)
[![Semantic Versions](https://img.shields.io/badge/%F0%9F%9A%80-semantic%20versions-informational.svg)](https://github.com/lgmoneda/time-robust-forest/releases)
[![License](https://img.shields.io/github/license/lgmoneda/time-robust-forest)](https://github.com/lgmoneda/time-robust-forest/blob/main/LICENSE)

</div>

A Proof of concept model that explores timestamp information to train a random forest with better Out of Distribution generalization power.

## Installation

```bash
pip install -U time-robust-forest
```

## How to use it

There are a classifier and a regressor under `time_robust_forest.models`. They follow the sklearn interface, which means you can quickly fit and use a model:

```python
from time_robust_forest.models import TimeForestClassifier

features = ["x_1", "x_2"]
time_column = "periods"
target = "y"

model = TimeForestClassifier(time_column=time_column)

model.fit(training_data[features + [time_column]], training_data[target])
predictions = model.predict_proba(test_data[features])[:, 1]
```

There are only a few arguments that differ from a traditional Random Forest. two arguments

- time_column: the column from the input dataframe containing the time
periods the model will iterate over to find the best splits (default: "period")
- min_sample_periods: the number of examples in every period the model needs
to keep while it splits.
- period_criterion: how the performance in every period is going to be
aggregated. Options: {"avg": average, "max": maximum, the worst case}.
(default: "avg")

To use the environment-wise optimization:

```python 
from time_robust_forest.hyper_opt import env_wise_hyper_opt

params_grid = {"n_estimators": [30, 60, 120], 
              "max_depth": [5, 10],
              "min_impurity_decrease": [1e-1, 1e-3, 0],
              "min_sample_periods": [5, 10, 30],
              "period_criterion": ["max", "avg"]}
			  
model = TimeForestClassifier(time_column=TIME_COLUMN)
										
opt_param = env_wise_hyper_opt(train[features + [TIME_COLUMN]], 
                               train[TARGET], 
                               model, 
                               TIME_COLUMN,
                               params_grid,
                               cv=5,
                               score=roc_auc_score)

```

### Make sure you have a good choice for the time column

Don't simply use a timestamp column from the dataset, make it discrete before and guarantee there is a reasonable amount of data points in every period. Example: use year if you have 3+ years of data. Notice the choice to make it discrete becomes a modeling choice you can optimize.

## License

[![License](https://img.shields.io/github/license/lgmoneda/time-robust-forest)](https://github.com/lgmoneda/time-robust-forest/blob/main/LICENSE)

This project is licensed under the terms of the `BSD-3` license. See [LICENSE](https://github.com/lgmoneda/time-robust-forest/blob/main/LICENSE) for more details.

## Useful links 

- [Introducing the Time Robust Tree blog post](http://lgmoneda.github.io/2021/12/03/introducing-time-robust-tree.html)

## Citation

```
@misc{time-robust-forest,
  author = {Moneda, Luis},
  title = {Time Robust Forest model},
  year = {2021},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/lgmoneda/time-robust-forest}}
}
```
