import numpy as np
import pandas as pd

from skscope.skmodel import PortfolioSelection 
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit


def test_portfolio():
    # load data
    port = PortfolioSelection(s=50, alpha=0.001, random_state=0)
    dir = "../docs/source/userguide/examples/Miscellaneous/data/csi500-2020-2021.csv"
    X = pd.read_csv(dir, encoding='gbk')
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
    param_grid = {'alpha':[1e-4, 1e-3, 1e-2]}
    port = PortfolioSelection(obj="MeanVar", s=50, random_state=0)
    grid_search = GridSearchCV(port, param_grid, cv=tscv)
    grid_search.fit(X)
    grid_search.cv_results_
    assert grid_search.best_score_ > 0.05
 
test_portfolio()
