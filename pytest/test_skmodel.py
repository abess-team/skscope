import os
import numpy as np
import pandas as pd

from skscope.skmodel import (
    PortfolioSelection,
    NonlinearSelection,
    RobustRegression,
    MultivariateFailure,
    IsotonicRegression,
)
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.estimator_checks import check_estimator
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit, train_test_split
from sklearn.feature_selection import SelectFromModel
from sklearn.neural_network import MLPRegressor

# from sksurv.datasets import load_veterans_lung_cancer
# from sksurv.preprocessing import OneHotEncoder
import warnings

warnings.filterwarnings("ignore")

CURRENT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_PortfolioSelection():
    # load data
    port = PortfolioSelection(sparsity=50, alpha=0.001, random_state=0)
    dir = os.path.normpath(
        "/docs/source/gallery/Miscellaneous/data/csi500-2020-2021.csv"
    )
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
    print("PortfolioSelection passed test!")


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
    print("NonlinearSelection passed test!")


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
    print("RobustRegression passed test!")


test_RobustRegression()


def test_MultivariateFailure():
    def make_Clayton2_data(n, theta=15, lambda1=1, lambda2=1, c1=1, c2=1):
        u1 = np.random.uniform(0, 1, n)
        u2 = np.random.uniform(0, 1, n)
        time2 = -np.log(1 - u2) / lambda2
        time1 = (
            np.log(
                1
                - np.power((1 - u2), -theta)
                + np.power((1 - u1), -theta / (1 + theta)) * np.power((1 - u2), -theta)
            )
            / theta
            / lambda1
        )
        ctime1 = np.random.uniform(0, c1, n)
        ctime2 = np.random.uniform(0, c2, n)
        delta1 = (time1 < ctime1) * 1
        delta2 = (time2 < ctime2) * 1
        censoringrate1 = 1 - sum(delta1) / n
        censoringrate2 = 1 - sum(delta2) / n
        # print("censoring rate1:" + str(censoringrate1))
        # print("censoring rate2:" + str(censoringrate2))
        time1 = np.minimum(time1, ctime1)
        time2 = np.minimum(time2, ctime2)
        y = np.hstack((time1.reshape((-1, 1)), time2.reshape((-1, 1))))
        delta = np.hstack((delta1.reshape((-1, 1)), delta2.reshape((-1, 1))))
        return y, delta

    K = 2
    np.random.seed(1234)
    n, p, s, rho = 100, 100, 10, 0.5
    beta = np.zeros(p)
    beta[:s] = 5
    Sigma = np.power(
        rho, np.abs(np.linspace(1, p, p) - np.linspace(1, p, p).reshape(p, 1))
    )
    X = np.random.multivariate_normal(mean=np.zeros(p), cov=Sigma, size=(n,))
    lambda1 = 1 * np.exp(np.matmul(X, beta))
    lambda2 = 10 * np.exp(np.matmul(X, beta))

    y, delta = make_Clayton2_data(
        n, theta=50, lambda1=lambda1, lambda2=lambda2, c1=5, c2=5
    )

    model = MultivariateFailure(s)
    model = model.fit(X, y, delta)
    assert (np.nonzero(model.coef_)[0] == np.nonzero(beta)[0]).all()
    pred = model.predict(X)
    score = model.score(X, y, delta)
    print("MultivariateFailure passed test!")


test_MultivariateFailure()

def test_IsotonicRegression():
    # check_estimator(IsotonicRegression())
    np.random.seed(0)
    n = 200
    X = np.arange(n) + 1
    y = 2 * np.log1p(np.arange(n)) + np.random.normal(size=n)
    model = IsotonicRegression(sparsity=10)
    model = model.fit(X, y)
    score = model.score(X, y)
    assert score >= 0.8
    X_new = model.transform(X)
    print("IsotonicRegression passed test!")

test_IsotonicRegression()
