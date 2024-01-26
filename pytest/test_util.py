import pytest
import numpy as np
from skscope.utilities import check_y_survival, check_array_survival


def test_check_y_survival():
    y = np.array([(True, 1.0), (False, 2.0)], dtype=[("event", bool), ("time", float)])
    event, time = check_y_survival(y)
    assert np.array_equal(event, np.array([True, False]))
    assert np.array_equal(time, np.array([1.0, 2.0]))

    event = np.array([True, False])
    time = np.array([1.0, 2.0])
    result = check_y_survival(event, time)
    assert np.array_equal(result[0], event)
    assert np.array_equal(result[1], time)

    with pytest.raises(ValueError):
        check_y_survival(np.array([1, 2, 3]))


def test_check_array_survival():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([(True, 1.0), (False, 2.0)], dtype=[("event", bool), ("time", float)])
    event, time = check_array_survival(X, y)
    assert np.array_equal(event, np.array([True, False]))
    assert np.array_equal(time, np.array([1.0, 2.0]))

    X = np.array([[1, 2], [3, 4], [5, 6]])
    with pytest.raises(ValueError):
        check_array_survival(X, y)
