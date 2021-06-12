import pdb
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn import metrics
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from time_robust_forest.functions import (
    check_min_sample_periods,
    check_min_sample_periods_dict,
    fill_right_dict,
    initialize_period_dict,
    score_by_period,
)


class TimeForestRegressor(BaseEstimator, RegressorMixin):
    """
        Time Forest Regressor Estimator.

        Arguments:
        - n_estimators: number of estimators to compose the ensemble (default: 5)
        - time_column: the column from the input dataframe containing the time
        periods the model will iterate over to find the best splits (default: "period")
        - max_depth: the maximum depth the trees are enabled to split (default: 5)
        - min_leaf: minimum number of total examples to split. It doesn't make any
    difference in cases min_leaf < |time_periods| * min_sample_periods (default: 5)
        - min_sample_periods: the number of examples in every period the model needs
        to keep while it splits.
        - max_features: the maximum number of features to be considered in a split.
        It is a fraction, so 1.0 is equivalent to use all the features. The default
        uses a common heuristic to define it(default: "auto")
        - bootstrapping: to perform bootstrapping before providing the input data to
        every estimator (default: True)
        - period_criterion: how the performance in every period is going to be
        aggregated. Options: {"avg": average, "max": maximum, the worst case}.
        (default: "avg")
        - n_jobs: number of cores to use when the parameter multi=True
        - multi: boolean to learn or not the many estimators in parallel
        (default: True)
    """

    def __init__(
        self,
        n_estimators=5,
        time_column="period",
        max_depth=5,
        min_leaf=5,
        min_sample_periods=100,
        max_features="auto",
        bootstrapping=True,
        period_criterion="avg",
        n_jobs=-1,
        multi=True,
    ):
        self.min_leaf, self.max_depth = min_leaf, max_depth
        self.time_column = time_column
        self.min_sample_periods = min_sample_periods
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.multi = multi
        self.bootstrapping = bootstrapping
        self.period_criterion = period_criterion

    def fit(self, X, y, sample_weight=None, verbose=False):
        """
        Learns the regressor model from the training data.

        - X: pd.DataFrame containing the training data, which include the
        features and the time_column informed in the model constructor.
        - y: pd.Series with the target variable (continuous).
        - sample_weight: a weight to consider in the loss function. It was
        implemented to enable boosting also.
        - verbose: If True, it is going to display the splits and how many
        examples by period every noode has after spliting.
        """
        if self.n_jobs <= 0:
            self.n_jobs = cpu_count() - 2

        if type(X) == np.ndarray:
            X = pd.DataFrame(X)
            X.columns = [*X.columns[:-1], self.time_column]
        if type(y) == pd.Series:
            y = y.values

        self.train_target_proportion = np.mean(y)
        self.classes_ = np.unique(y)

        self.n_estimators_ = []
        if not self.multi:
            self.n_estimators_ = [
                _RandomTimeSplitTree(
                    X,
                    y,
                    min_sample_periods=self.min_sample_periods,
                    max_depth=self.max_depth,
                    min_leaf=self.min_leaf,
                    bootstrapping=self.bootstrapping,
                    sample_weight=sample_weight,
                    time_column=self.time_column,
                    row_indexes=[],
                    verbose=verbose,
                    max_features=self.max_features,
                    criterion="std",
                    period_criterion=self.period_criterion,
                    random_state=i,
                )
                for i in range(self.n_estimators)
            ]
        else:
            self.n_estimators_ = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(_RandomTimeSplitTree)(
                    X,
                    y,
                    min_sample_periods=self.min_sample_periods,
                    max_depth=self.max_depth,
                    min_leaf=self.min_leaf,
                    bootstrapping=self.bootstrapping,
                    sample_weight=sample_weight,
                    time_column=self.time_column,
                    row_indexes=[],
                    verbose=verbose,
                    max_features=self.max_features,
                    period_criterion=self.period_criterion,
                    criterion="std",
                    random_state=i,
                )
                for i in range(self.n_estimators)
            )

    def predict(self, X):
        """
        Predicts the output from the X input.

        - X: pd.Dataframe containing the input features.
        ---
        - average_prediction: outputs the average prediction from the
        n_estimators.
        """
        predictions = [model.predict(X) for model in self.n_estimators_]

        if type(X).__module__ == np.__name__:
            X = pd.DataFrame(X)

        predictions = [model.predict(X) for model in self.n_estimators_]
        average_prediction = np.mean(np.array(predictions), axis=0)

        return average_prediction

    def score(self, X, y):
        """
        Scores the quality of the model using mean squared error.
        """
        predictions = self.predict_proba_(X)
        return metrics.mean_squared_error(y, predictions)

    def feature_importance(self):
        """
        Retrieves the feature importance in terms of number of
        times a feature was used to split the data.

        It returns a ordered dataframe with feature names and number of splits.
        """
        return (
            pd.concat(
                [
                    n_estimator.feature_importance
                    for n_estimator in self.n_estimators_
                ]
            )
            .groupby(level=0)
            .sum()
            .sort_values(ascending=False)
        )


class TimeForestClassifier(BaseEstimator, ClassifierMixin):
    """
        Time Forest Classifier Estimator.

        Arguments:
        - n_estimators: number of estimators to compose the ensemble (default: 5)
        - time_column: the column from the input dataframe containing the time
        periods the model will iterate over to find the best splits (default: "period")
        - max_depth: the maximum depth the trees are enabled to split (default: 5)
        - min_leaf: minimum number of total examples to split. It doesn't make any
    difference in cases min_leaf < |time_periods| * min_sample_periods (default: 5)
        - min_sample_periods: the number of examples in every period the model needs
        to keep while it splits.
        - max_features: the maximum number of features to be considered in a split.
        It is a fraction, so 1.0 is equivalent to use all the features. The default
        uses a common heuristic to define it(default: "auto")
        - bootstrapping: to perform bootstrapping before providing the input data to
        every estimator (default: True)
        - period_criterion: how the performance in every period is going to be
        aggregated. Options: {"avg": average, "max": maximum, the worst case}.
        (default: "avg")
        - n_jobs: number of cores to use when the parameter multi=True
        - multi: boolean to learn or not the many estimators in parallel
        (default: True)
    """

    def __init__(
        self,
        n_estimators=5,
        time_column="period",
        max_depth=5,
        min_leaf=5,
        min_sample_periods=100,
        max_features="auto",
        bootstrapping=True,
        period_criterion="avg",
        n_jobs=-1,
        multi=True,
    ):
        self.min_leaf, self.max_depth = min_leaf, max_depth
        self.time_column = time_column
        self.min_sample_periods = min_sample_periods
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.n_jobs = n_jobs
        self.multi = multi
        self.bootstrapping = bootstrapping
        self.period_criterion = period_criterion

    def fit(self, X, y, sample_weight=None, verbose=False):
        """
        Learns the classifier model from the training data.

        - X: pd.DataFrame containing the training data, which include the
        features and the time_column informed in the model constructor.
        - y: pd.Series with the target variable (it should be binary).
        - sample_weight: a weight to consider in the loss function. It was
        implemented to enable boosting also.
        - verbose: If True, it is going to display the splits and how many
        examples by period every noode has after spliting.
        """
        if self.n_jobs <= 0:
            self.n_jobs = cpu_count() - 2

        if type(X) == np.ndarray:
            X = pd.DataFrame(X)
            X.columns = [*X.columns[:-1], self.time_column]
        if type(y) == pd.Series:
            y = y.values

        self.train_target_proportion = np.mean(y)
        self.classes_ = np.unique(y)

        self.n_estimators_ = []
        if not self.multi:
            self.n_estimators_ = [
                _RandomTimeSplitTree(
                    X,
                    y,
                    min_sample_periods=self.min_sample_periods,
                    max_depth=self.max_depth,
                    min_leaf=self.min_leaf,
                    bootstrapping=self.bootstrapping,
                    sample_weight=sample_weight,
                    time_column=self.time_column,
                    row_indexes=[],
                    verbose=verbose,
                    max_features=self.max_features,
                    criterion="gini",
                    period_criterion=self.period_criterion,
                    random_state=i,
                )
                for i in range(self.n_estimators)
            ]
        else:
            self.n_estimators_ = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(_RandomTimeSplitTree)(
                    X,
                    y,
                    min_sample_periods=self.min_sample_periods,
                    max_depth=self.max_depth,
                    min_leaf=self.min_leaf,
                    bootstrapping=self.bootstrapping,
                    sample_weight=sample_weight,
                    time_column=self.time_column,
                    row_indexes=[],
                    verbose=verbose,
                    max_features=self.max_features,
                    period_criterion=self.period_criterion,
                    criterion="gini",
                    random_state=i,
                )
                for i in range(self.n_estimators)
            )

    def predict_proba(self, X):
        """
        Predicts the likelihood of the negative and positive case given
        the input features.

        It returns the average results from all the n_estimators. It follows
        sklearn interface and provide a n_samples rows and two columns. To get
        the positive likelihood, use predicted_proba[:, 1].
        """
        if type(X).__module__ == np.__name__:
            X = pd.DataFrame(X)

        predictions = [model.predict(X) for model in self.n_estimators_]
        positive_proba = np.mean(np.array(predictions), axis=0)
        negative_proba = np.ones(len(positive_proba)) - positive_proba

        return np.vstack([negative_proba, positive_proba]).transpose()

    def predict_proba_(self, X):
        """
        Literally a function I did due to laziness about getting the
        positive case likelihood.
        """
        return self.predict_proba(X)[:, 1]

    def predict(self, X):
        """
        Predicts the class  given the input features.
        It returns the binary class.
        """
        predictions = self.predict_proba_(X)
        predictions = (predictions >= self.train_target_proportion) * 1
        return predictions

    def score(self, X, y):
        """
        Scores the quality of the model using roc auc score.
        """
        predictions = self.predict_proba_(X)
        return metrics.roc_auc_score(y, predictions)

    def feature_importance(self):
        """
        Retrieves the feature importance in terms of number of
        times a feature was used to split the data.

        It returns a ordered dataframe with feature names and number of splits.
        """
        return (
            pd.concat(
                [
                    n_estimator.feature_importance
                    for n_estimator in self.n_estimators_
                ]
            )
            .groupby(level=0)
            .sum()
            .sort_values(ascending=False)
        )


class _RandomTimeSplitTree:
    """
        THe basic class to split data recursively. It's used both for the regressor
        and classifier. It can be used to learn the Time Robust Tree directly.

        Arguments:
        - X: pandas DataFrame containing the input features and the time_column.
        - y: pandas Series or numpy array containing the target.
        - row_indexes: the indexes from the data we should consider to learn the
        next split, it's an internal variable to enable the recursion passing to the
        next call the entire data and indicating the valid indexes.
        - n_estimators: number of estimators to compose the ensemble (default: 5)
        - time_column: the column from the input dataframe containing the time
        periods the model will iterate over to find the best splits (default: "period")
        - max_depth: the maximum depth the trees are enabled to split (default: 5)
        - min_leaf: minimum number of total examples to split. It doesn't make any
    difference in cases min_leaf < |time_periods| * min_sample_periods (default: 5)
        - min_sample_periods: the number of examples in every period the model needs
        to keep while it splits.
        - max_features: the maximum number of features to be considered in a split.
        It is a fraction, so 1.0 is equivalent to use all the features. The default
        uses a common heuristic to define it(default: "auto")
        - bootstrapping: to perform bootstrapping before providing the input data to
        every estimator (default: True)
        - criterion: score function to be used when evaluating splits.
        {"gini": Gini Impurity, "std": Standard deviation reduction}
        (default: "gini").
        - split_verbose: to print splits evaluation. It's only recommended for
        checking pocket examples (default: False).
        - verbose: shows how many example there are in every period after
        every split (default: False).
        - random_state: random seed for the bootstrapping (default: 42).

    """

    def __init__(
        self,
        X,
        y,
        row_indexes=[],
        time_column="period",
        max_depth=5,
        min_leaf=5,
        max_features="auto",
        bootstrapping=True,
        criterion="gini",
        period_criterion="avg",
        min_sample_periods=100,
        sample_weight=None,
        depth=None,
        verbose=False,
        split_verbose=False,
        random_state=42,
    ):
        if len(row_indexes) == 0:
            row_indexes = np.arange(len(y))
        ### Reindex
        if depth == None:
            depth = 0
            if bootstrapping:
                resampled_X = X.sample(
                    frac=1.0, replace=True, random_state=random_state
                )
                resampled_idx = resampled_X.index
                X = resampled_X
                X.reset_index(inplace=True, drop=True)
                if type(y) == pd.DataFrame:
                    y = y.values
                elif type(y) == pd.Series:
                    y = y.values
                y = y[resampled_idx]

        self.X, self.y, self.row_indexes, self.max_depth = (
            X,
            y,
            row_indexes,
            max_depth,
        )
        self.min_leaf, self.depth = min_leaf, depth
        self.time_column = time_column
        self.min_sample_periods = min_sample_periods
        self.verbose = verbose
        self.split_verbose = split_verbose
        self.max_features = max_features
        self.split_variable = "LEAF"
        self.bootstrapping = bootstrapping
        self.criterion = criterion
        self.period_criterion = period_criterion

        if sample_weight is not None:
            self.sample_weight = sample_weight
        else:
            self.sample_weight = np.ones(len(y))

        self.n_examples = len(row_indexes)
        self.variables = [col for col in X.columns if col != time_column]
        if max_features == "auto":
            self.max_n_variables = max(int(len(self.variables) ** 0.5), 1)
        else:
            self.max_n_variables = max(
                int(max_features * len(self.variables)), 1
            )

        self.value = np.mean(y[row_indexes])
        self.score = float("inf")
        if verbose:
            print(f"Depth: {self.depth}")
            print(f"Max Depth: {self.max_depth}")
            print("Node periods distribution")
            print(
                self.X.loc[self.row_indexes, self.time_column]
                .value_counts()
                .sort_index()
            )

        if self.depth == 0:
            if (
                check_min_sample_periods(
                    self.X.loc[self.row_indexes],
                    self.time_column,
                    self.min_sample_periods,
                )
                == 0
            ):
                print(
                    "Not enough sample in the periods to perform"
                    + "a split using {} as minimum sample by period".format(
                        min_sample_periods
                    )
                )
        elif self.depth <= self.max_depth:
            self.create_split()

    def create_split(self):
        """
        Selects a subset of the input features (when the parameter
        max_features enables it), look for the best feature and value to split
        the data by calling the function to find the best split for every feature
        considering this chosen set, perform the split and make the
        recursive call to build sub tress using the result splits.
        """
        variables_to_consider = np.random.choice(
            self.variables, self.max_n_variables, replace=False
        )
        for idx, variable in enumerate(self.variables):
            if variable in variables_to_consider:
                self.find_better_split(variable, idx)
        if self.score == float("inf"):
            return False
        x = self._split_column

        left_split = np.nonzero(x <= self.split_example)
        right_split = np.nonzero(x > self.split_example)

        self.left_split = _RandomTimeSplitTree(
            self.X,
            self.y,
            self.row_indexes[left_split],
            depth=self.depth + 1,
            max_features=self.max_features,
            bootstrapping=self.bootstrapping,
            min_sample_periods=self.min_sample_periods,
            time_column=self.time_column,
            max_depth=self.max_depth,
            sample_weight=self.sample_weight,
            verbose=self.verbose,
        )
        self.right_split = _RandomTimeSplitTree(
            self.X,
            self.y,
            self.row_indexes[right_split],
            depth=self.depth + 1,
            max_features=self.max_features,
            bootstrapping=self.bootstrapping,
            min_sample_periods=self.min_sample_periods,
            time_column=self.time_column,
            max_depth=self.max_depth,
            sample_weight=self.sample_weight,
            verbose=self.verbose,
        )

    def find_better_split(self, variable, variable_idx):
        """
        Given an input feature variable, it finds the best split possible
        using it. If it is better than the current stored split, it replaces
        it by the current variable and the best split form it.
        """
        x, y = self.X.loc[self.row_indexes, variable], self.y[self.row_indexes]
        weights = self.sample_weight[self.row_indexes]

        #### Check for the minimum number of examples in every period
        period_data = self.X.loc[self.row_indexes, self.time_column]
        unique_periods = period_data.unique()
        x = x.values

        sorted_indexes = np.argsort(x)
        sorted_x, sorted_y = x[sorted_indexes], y[sorted_indexes]
        sorted_weights = weights[sorted_indexes]

        sorted_period_data = period_data.iloc[sorted_indexes]
        right_periods_count = sorted_period_data.value_counts().to_dict()
        left_periods_count = {key: 0 for key in right_periods_count.keys()}
        right_period_dict = initialize_period_dict(unique_periods)
        left_period_dict = initialize_period_dict(unique_periods)

        right_period_dict = fill_right_dict(
            sorted_period_data, sorted_y, sorted_weights, right_period_dict
        )

        for example in range(0, self.n_examples - self.min_leaf - 1):
            x_i, y_i = sorted_x[example], sorted_y[example]
            period_i = sorted_period_data.iloc[example]
            weight_i = sorted_weights[example]

            right_periods_count[period_i] -= 1
            left_periods_count[period_i] += 1

            ### Update every period stats
            right_period_dict[period_i]["count"] -= weight_i
            left_period_dict[period_i]["count"] += weight_i
            right_period_dict[period_i]["sum"] -= y_i * weight_i
            left_period_dict[period_i]["sum"] += y_i * weight_i

            if self.criterion == "std":
                right_period_dict[period_i]["squared_sum"] -= (
                    y_i ** 2
                ) * weight_i
                left_period_dict[period_i]["squared_sum"] += (
                    y_i ** 2
                ) * weight_i

            if example < self.min_leaf or x_i == sorted_x[example + 1]:
                continue
            elif not check_min_sample_periods_dict(
                right_periods_count, self.min_sample_periods
            ) or not check_min_sample_periods_dict(
                left_periods_count, self.min_sample_periods
            ):
                continue
            if not check_min_sample_periods_dict(
                right_periods_count, self.min_sample_periods
            ):
                break

            if self.split_verbose:
                print(f"Evaluate a split on variable {variable} at value {x_i}")
            current_score = score_by_period(
                right_period_dict,
                left_period_dict,
                self.criterion,
                self.period_criterion,
                self.split_verbose,
            )

            if current_score < self.score:
                self.split_variable, self.score, self.split_example = (
                    variable,
                    current_score,
                    x_i,
                )
                self.split_variable_idx = variable_idx

    def _is_leaf(self):
        """
        Returns if the current instance of the tree is a leaf.
        """
        return self.score == float("inf")

    def _split_column(self):
        """
        Returns what is the current vector holding the best split.
        """
        return self.X.values[self.row_indexes, self.split_variable_idx]

    def predict(self, X):
        """
        Predicts

        - X: pd.DataFrame containing all the input features.
        """
        if type(X) == np.ndarray:
            X = pd.DataFrame(X, columns=self.variables + [self.time_column])
        return np.array([self._predict_row(x) for x in X.iterrows()])

    def _predict_row(self, x):
        """
        Auxiliary function to predict a single row recursively.
        """
        if self._is_leaf():
            return self.value
        tree = (
            self.left_split
            if x[1][self.split_variable] <= self.split_example
            else self.right_split
        )

        return tree._predict_row(x)

    def _get_split_variable(self):
        """
        Returns the splitting variable name for the current tree instance.
        """
        if not self._is_leaf():
            return (
                self.split_variable
                + "@"
                + self.left_split._get_split_variable()
                + "@"
                + self.right_split._get_split_variable()
            )
        return "LEAF"

    def feature_importance(self):
        """
        Retrieves the feature importance in terms of number of
        times a feature was used to split the data.

        It returns a ordered dataframe with feature names and number of splits.
        """
        splits = self._get_split_variable()
        splits_features = splits.replace("@LEAF", "").split("@")

        return (
            pd.DataFrame(splits_features, columns=["Feature Importance"])[
                "Feature Importance"
            ]
            .value_counts()
            .sort_values(ascending=False)
        )
