from functools import partial

import pandas as pd
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold


def extract_results_from_grid_cv(cv_results, kfolds, envs):
    """
    Extract the resuls from a fitted grid search object from sklearn
    to enable picking the best using custom logic.
    """

    split_keys = [i for i in cv_results.keys() if "split" in i]

    split_env = {
        split_key: split_key.split("env_")[-1]
        for i, split_key in enumerate(split_keys)
    }
    params_idx = [i for i in range(len(cv_results["params"]))]
    all_folds_df = []
    for kfold, split_key in enumerate(split_keys):
        fold_df = pd.DataFrame()

        fold_df["perf"] = cv_results[split_key]
        fold_df["split"] = kfold
        fold_df["env"] = split_env[split_key]
        fold_df["params"] = cv_results["params"]
        fold_df["params_idx"] = params_idx
        all_folds_df.append(fold_df)

    results_df = pd.concat(all_folds_df)

    return results_df


def select_best_model_from_results_df(results_df):
    """
    Aggregates the result df extracted from a fitted grid search to return
    the best parameters.
    """

    first_agg_dict = {"params": "first", "perf": "mean"}
    second_agg_dict = {"params": "first", "perf": "min"}

    results_df = results_df.groupby(["params_idx", "env"], as_index=False).agg(
        first_agg_dict
    )
    results_df = results_df.groupby("params_idx").agg(second_agg_dict)

    return results_df.iloc[results_df["perf"].argmax()]["params"], results_df


def env_stratified_folds(data, env_column="period", cv=5):
    """
    Create folds that are stratified on the environment.
    """
    envs = data[env_column].unique()
    cv_sets = []
    kfolds = StratifiedKFold(n_splits=cv)
    for train_idx, test_idx in kfolds.split(data, data[env_column]):
        cv_sets.append((train_idx, test_idx))

    return cv_sets


def env_wise_score(estimator, X, y, scorer, env, env_column):
    """
    Filter data to evaluate only a specific environment using a
    certain scorer.
    """
    env_mask = X[env_column] == env
    evaluation = scorer(estimator, X[env_mask], y[env_mask])

    return evaluation


def grid_search(X, y, model, param_grid, env_cvs, scorer):
    """
    FIt the grid search and return it.
    """

    grid_cv = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=env_cvs,
        scoring=scorer,
        n_jobs=-1,
        verbose=0,
        refit=False,
    )

    grid_cv.fit(X, y)
    return grid_cv


def env_wise_hyper_opt(
    X,
    y,
    model,
    env_column,
    param_grid,
    cv=5,
    scorer=make_scorer(roc_auc_score, needs_proba=True),
    ret_results=False,
):
    """
    Optimize the hyper parmaters of a model considering the leave one env out
    cross-validation and selecting the worst case regarding the test performance
    in the different environments.
    """
    env_cvs = env_stratified_folds(X, env_column, cv)
    envs = X[env_column].unique()

    scoring_fs = {
        f"{scorer.__repr__()}_env_{env}": partial(
            env_wise_score, scorer=scorer, env=env, env_column=environment
        )
        for env in envs
    }

    grid_cv = grid_search(X, y, model, param_grid, env_cvs, scoring_fs)

    results_df = extract_results_from_grid_cv(grid_cv.cv_results_, cv, envs)

    opt_params, agg_results_df = select_best_model_from_results_df(results_df)

    if ret_results:
        return opt_params, results_df, agg_results_df

    return opt_params
