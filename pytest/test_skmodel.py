import os
import numpy as np
import pandas as pd

from skscope.skmodel import PortfolioSelection, NonlinearSelection
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

CURRENT = os.path.dirname(os.path.abspath(__file__))


def test_portfolio():
    # load data
    port = PortfolioSelection(s=50, alpha=0.001, random_state=0)
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
    port = PortfolioSelection(obj="MeanVar", s=50, random_state=0)
    grid_search = GridSearchCV(port, param_grid, cv=tscv)
    grid_search.fit(X)
    grid_search.cv_results_
    assert grid_search.best_score_ > 0.05


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

    selector = NonlinearSelection(5)
    selector.fit(X, y)
    assert set(np.nonzero(selector.coef_)[0]) == set(true_support_set)
