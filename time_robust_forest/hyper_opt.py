import pandas as pd
from sklearn.metrics import make_scorer, roc_auc_score
from sklearn.model_selection import GridSearchCV, KFold


def extract_results_from_grid_cv(cv_results, kfolds, envs):
    """
    Extract the resuls from a fitted grid search object from sklearn so
    """
    split_keys = [i for i in cv_results.keys() if "split" in i]

    split_env = {
        split_key: envs[i % len(envs)] for i, split_key in enumerate(split_keys)
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

    return results_df.iloc[results_df["perf"].argmax()]["params"]


def leave_one_env_out_cv(data, env_column="period", cv=5):
    """
    Create folds that keep only one environment in the test fold.
    """
    envs = data[env_column].unique()
    cv_sets = []
    kfolds = KFold(n_splits=cv)
    for train_idx, test_idx in kfolds.split(data):
        for env in envs:
            all_env_elements = data[data[env_column] == env].index
            test_env_idx = [i for i in test_idx if i in all_env_elements]
            cv_sets.append((train_idx, test_env_idx))

    return cv_sets


def grid_search(X, y, model, param_grid, env_cvs, score):
    """
    FIt the grid search and return it.
    """

    grid_cv = GridSearchCV(
        model,
        param_grid=param_grid,
        cv=env_cvs,
        scoring=make_scorer(score),
        n_jobs=-1,
        verbose=0,
    )

    grid_cv.fit(X, y)
    return grid_cv


def env_wise_hyper_opt(
    X, y, model, env_column, param_grid, cv=5, score=roc_auc_score
):
    """
    Optimize the hyper parmaters of a model considering the leave one env out
    cross-validation and selecting the worst case regarding the test performance
    in the different environments.
    """
    env_cvs = leave_one_env_out_cv(X, env_column, cv)

    grid_cv = grid_search(X, y, model, param_grid, env_cvs, score)

    envs = X[env_column].unique()
    results_df = extract_results_from_grid_cv(grid_cv.cv_results_, cv, envs)

    opt_params = select_best_model_from_results_df(results_df)

    return opt_params
