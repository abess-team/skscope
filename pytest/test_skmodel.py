import os
import numpy as np
import pandas as pd

from skscope.skmodel import (
    PortfolioSelection,
    NonlinearSelection,
    RobustRegression,
    CoxPH
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPRegressor
from sksurv.datasets import load_veterans_lung_cancer
from sksurv.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings("ignore")

CURRENT = os.path.dirname(os.path.abspath(__file__))


def test_PortfolioSelection():
    # load data
    port = PortfolioSelection(sparsity=50, alpha=0.001, random_state=0)
    dir = "/../docs/source/userguide/examples/Miscellaneous/data/csi500-2020-2021.csv"
    X = pd.read_csv(CURRENT + dir, encoding="gbk")
    keep_cols = X.columns[(X.isnull().sum() <= 20)]
    X = X[keep_cols]
    X = X.fillna(0)
    X = X.iloc[:, 1:].values / 100

    # train-test splitting
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    X_test = X[train_size:]

    # fit and test
    port = port.fit(X_train)
    score = port.score(X_test)
    assert score > 0.05

    # gridsearch with time-series splitting
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = {"alpha": [1e-4, 1e-3, 1e-2]}
    port = PortfolioSelection(obj="MeanVar", sparsity=50, random_state=0)
    grid_search = GridSearchCV(port, param_grid, cv=tscv)
    grid_search.fit(X)
    grid_search.cv_results_
    assert grid_search.best_score_ > 0.05


test_PortfolioSelection()


def test_NonlinearSelection():
    n = 1000
    p = 10
    sparsity_level = 5
    rng = np.random.default_rng(100)
    X = rng.normal(0, 1, (n, p))
    noise = rng.normal(0, 1, n)

    true_support_set = rng.choice(np.arange(p), sparsity_level, replace=False)
    true_support_set_list = np.split(true_support_set, 5)

    y = (
        np.sum(
            X[:, true_support_set_list[0]] * np.exp(2 * X[:, true_support_set_list[1]]),
            axis=1,
        )
        + np.sum(np.square(X[:, true_support_set_list[2]]), axis=1)
        + np.sum(
            (2 * X[:, true_support_set_list[3]] - 1)
            * (2 * X[:, true_support_set_list[4]] - 1),
            axis=1,
        )
        + noise
    )

    check_estimator(NonlinearSelection())
    selector = NonlinearSelection(5)
    selector.fit(X, y)
    assert set(np.nonzero(selector.coef_)[0]) == set(true_support_set)

    # supervised dimension reduction
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)

    mlp = MLPRegressor(random_state=1, max_iter=1000).fit(X_train, y_train)
    score1 = mlp.score(X_test, y_test)

    selector = NonlinearSelection(5).fit(X_train, y_train)
    transformer = SelectFromModel(selector, threshold=1e-7, prefit=True)
    estimators = [
        ("reduce_dim", transformer),
        ("reg", MLPRegressor(random_state=1, max_iter=1000)),
    ]
    pipe = Pipeline(estimators).fit(X_train, y_train)
    score2 = pipe.score(X_test, y_test)
    assert score2 > score1

    # gridsearch
    # selector = NonlinearSelection(5)
    # param_grid = {"gamma_x": [0.7, 1.5], "gamma_y": [0.7, 1.5]}
    # grid_search = GridSearchCV(selector, param_grid)
    # grid_search.fit(X, y)
    # grid_search.cv_results_
    # assert set(np.nonzero(grid_search.best_estimator_.coef_)[0]) == set(true_support_set)


test_NonlinearSelection()


def test_RobustRegression():
    n = 1000
    p = 10
    sparsity_level = 5
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (n, p))
    noise = rng.standard_cauchy(n)

    true_support_set = rng.choice(np.arange(p), sparsity_level, replace=False)
    beta_true = np.zeros(p)
    beta_true[true_support_set] = 1
    y = X @ beta_true + noise

    check_estimator(RobustRegression())
    model = RobustRegression(sparsity=5, gamma=1)
    model = model.fit(X, y)
    sample_weight = rng.random(n)
    score = model.score(X, y, sample_weight)
    est_support_set = np.nonzero(model.coef_)[0]
    assert set(est_support_set) == set(true_support_set)


test_RobustRegression()

def test_CoxPH():
    data_x, data_y = load_veterans_lung_cancer()
    data_x_numeric = OneHotEncoder().fit_transform(data_x)
    X, y = data_x_numeric.values, data_y

    model = CoxPH(3)
    model = model.fit(X, y)
    pred = model.predict(X)
    score = model.score(X, y)

test_CoxPH()