import numpy as np
import pandas as pd

from skscope.skmodel import PortfolioSelection 


def test_portfolio():
    port = PortfolioSelection(k=50, seed=0)
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

    port = port.fit(X_train, obj="MinVar")
    score = port.score(X_test)
    assert score > 0.1
    return np.round(score, 3)

score = test_portfolio()

