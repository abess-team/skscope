import numpy as np
from sklearn.utils import check_array, check_consistent_length


def check_y_survival(y_or_event, *args, allow_all_censored=False):
    """Check that array correctly represents an outcome for survival analysis.

    Parameters
    ----------
    y_or_event : structured array with two fields, or boolean array
        If a structured array, it must contain the binary event indicator
        as first field, and time of event or time of censoring as
        second field. Otherwise, it is assumed that a boolean array
        representing the event indicator is passed.

    *args : list of array-likes
        Any number of array-like objects representing time information.
        Elements that are `None` are passed along in the return value.

    allow_all_censored : bool, optional, default: False
        Whether to allow all events to be censored.

    Returns
    -------
    event : array, shape=[n_samples,], dtype=bool
        Binary event indicator.

    time : array, shape=[n_samples,], dtype=float
        Time of event or censoring.
    """
    if len(args) == 0:
        y = y_or_event

        if (
            not isinstance(y, np.ndarray)
            or y.dtype.fields is None
            or len(y.dtype.fields) != 2
        ):
            raise ValueError(
                "y must be a structured array with the first field"
                " being a binary class event indicator and the second field"
                " the time of the event/censoring"
            )

        event_field, time_field = y.dtype.names
        y_event = y[event_field]
        time_args = (y[time_field],)
    else:
        y_event = np.asanyarray(y_or_event)
        time_args = args

    event = check_array(y_event, ensure_2d=False)
    if not np.issubdtype(event.dtype, np.bool_):
        raise ValueError(
            f"elements of event indicator must be boolean, but found {event.dtype}"
        )

    if not (allow_all_censored or np.any(event)):
        raise ValueError("all samples are censored")

    return_val = [event]
    for i, yt in enumerate(time_args):
        if yt is None:
            return_val.append(yt)
            continue

        yt = check_array(yt, ensure_2d=False)
        if not np.issubdtype(yt.dtype, np.number):
            raise ValueError(
                f"time must be numeric, but found {yt.dtype} for argument {i + 2}"
            )

        return_val.append(yt)

    return tuple(return_val)


def check_array_survival(X, y):
    """Check that all arrays have consistent first dimensions.

    Parameters
    ----------
    X : array-like
        Data matrix containing feature vectors.

    y : structured array with two fields
        A structured array containing the binary event indicator
        as first field, and time of event or time of censoring as
        second field.

    Returns
    -------
    event : array, shape=[n_samples,], dtype=bool
        Binary event indicator.

    time : array, shape=[n_samples,], dtype=float
        Time of event or censoring.
    """
    event, time = check_y_survival(y)
    check_consistent_length(X, event, time)
    return event, time


def AIC(
    objective_value: float,
    dimensionality: int,
    effective_params_num: int,
    train_size: int,
) -> float:
    """
    Calculate the Akaike Information Criterion (AIC) for a given model.

    Parameters:
    - objective_value (float): The value of the objective function (e.g., negative log-likelihood) for the model.
    - dimensionality (int): The number of dimensions (features) in the model.
    - effective_params_num (int): The number of effective parameters in the model.
    - train_size (int): The size of the training dataset.

    Returns:
    - float: The AIC value, which is a measure of the relative quality of the model for a given set of data.
      Lower AIC values indicate a better model.

    Note:
    When using information criteria (IC) functions, it is crucial to pay attention to different loss functions
    for the same model. The correct IC implementation may vary. This function assumes the loss function is the
    negative log-likelihood of the model. For example:

    def logistic_loss(params):
        xbeta = X @ params
        return jnp.sum(jnp.logaddexp(0, xbeta) - y * xbeta)

    If use `mean` rather than `sum` in `logistic_loss`, IC value may be wrong.
    The AIC is calculated using the formula:
    AIC = 2 * (objective_value + effective_params_num)
    """
    return 2 * objective_value + 2 * effective_params_num


def BIC(
    objective_value: float,
    dimensionality: int,
    effective_params_num: int,
    train_size: int,
):
    """
    Calculate the Bayesian Information Criterion (BIC) for a given model.

    Parameters:
    - objective_value (float): The value of the objective function (e.g., negative log-likelihood) for the model.
    - dimensionality (int): The number of dimensions (features) in the model.
    - effective_params_num (int): The number of effective parameters in the model.
    - train_size (int): The size of the training dataset.

    Returns:
    - float: The BIC value, which is a measure of the relative quality of the model for a given set of data.
      Lower BIC values indicate a better model.

    Note:
    When using information criteria (IC) functions, it is crucial to pay attention to different loss functions
    for the same model. The correct IC implementation may vary. This function assumes the loss function is the
    negative log-likelihood of the model. For example:

    def logistic_loss(params):
        xbeta = X @ params
        return jnp.sum(jnp.logaddexp(0, xbeta) - y * xbeta)

    If use `mean` rather than `sum` in `logistic_loss`, IC value may be wrong.
    The BIC is calculated using the formula:
    BIC = 2 * objective_value + effective_params_num * np.log(train_size)
    """
    return 2 * objective_value + effective_params_num * np.log(train_size)


def GIC(
    objective_value: float,
    dimensionality: int,
    effective_params_num: int,
    train_size: int,
):
    """
    Calculate the Generalized Information Criterion (GIC) for a given model.

    Parameters:
    - objective_value (float): The value of the objective function (e.g., negative log-likelihood) for the model.
    - dimensionality (int): The number of dimensions (features) in the model.
    - effective_params_num (int): The number of effective parameters in the model.
    - train_size (int): The size of the training dataset.

    Returns:
    - float: The GIC value, which is a measure of the relative quality of the model for a given set of data.
      Lower GIC values indicate a better model.

    Note:
    When using information criteria (IC) functions, it is crucial to pay attention to different loss functions
    for the same model. The correct IC implementation may vary. This function assumes the loss function is the
    negative log-likelihood of the model. For example:

    def logistic_loss(params):
        xbeta = X @ params
        return jnp.sum(jnp.logaddexp(0, xbeta) - y * xbeta)

    If use `mean` rather than `sum` in `logistic_loss`, IC value may be wrong.
    The GIC is calculated using the formula:
    GIC = 2 * objective_value + effective_params_num * np.log(np.log(train_size)) * np.log(dimensionality)
    """
    return 2 * objective_value + effective_params_num * np.log(
        np.log(train_size)
    ) * np.log(dimensionality)


def EBIC(
    objective_value: float,
    dimensionality: int,
    effective_params_num: int,
    train_size: int,
):
    """
    Calculate the Extended Bayesian Information Criterion (EBIC) for a given model.

    Parameters:
    - objective_value (float): The value of the objective function (e.g., negative log-likelihood) for the model.
    - dimensionality (int): The number of dimensions (features) in the model.
    - effective_params_num (int): The number of effective parameters in the model.
    - train_size (int): The size of the training dataset.

    Returns:
    - float: The EBIC value, which is a measure of the relative quality of the model for a given set of data.
      Lower EBIC values indicate a better model.

    Note:
    When using information criteria (IC) functions, it is crucial to pay attention to different loss functions
    for the same model. The correct IC implementation may vary. This function assumes the loss function is the
    negative log-likelihood of the model. For example:

    def logistic_loss(params):
        xbeta = X @ params
        return jnp.sum(jnp.logaddexp(0, xbeta) - y * xbeta)

    If use `mean` rather than `sum` in `logistic_loss`, IC value may be wrong.
    The E is calculated using the formula:
    EBIC = 2 * objective_value + effective_params_num * (np.log(train_size) + 2 * np.log(dimensionality))
    """
    return 2 * objective_value + effective_params_num * (
        np.log(train_size) + 2 * np.log(dimensionality)
    )


def LinearSIC(
    objective_value: float,
    dimensionality: int,
    effective_params_num: int,
    train_size: int,
):
    """
    Calculate the Special Information Criterion (SIC) for a linear model.

    The difference between GIC and LinearSIC:
    GIC assumes that the objective function is the negative logarithmic likelihood function of a statistical model;
    LinearSIC assumes that the objective function is the sum of squared residuals, specifically adapted to linear models.

    Parameters:
    - objective_value (float): The value of the objective function (e.g., negative log-likelihood) for the model.
    - dimensionality (int): The number of dimensions (features) in the model.
    - effective_params_num (int): The number of effective parameters in the model.
    - train_size (int): The size of the training dataset.

    Returns:
    - float: The LinearSIC value, which is a measure of the relative quality of the model for a given set of data.
      Lower LinearSIC values indicate a better model.

    Note:
    When using information criteria (IC) functions, it is crucial to pay attention to different loss functions
    for the same model. The correct IC implementation may vary. This function assumes the loss function is the
    negative log-likelihood of the model. For example:

    def linear_loss(params):
        return jnp.sum(jnp.square(y - X @ params))

    The LinearSIC is calculated using the formula:
    LinearSIC = train_size * np.log(objective_value) + 2 * effective_params_num * np.log(np.log(train_size)) * np.log(dimensionality)
    """
    return train_size * np.log(objective_value) + 2 * effective_params_num * np.log(
        np.log(train_size)
    ) * np.log(dimensionality)
