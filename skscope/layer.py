from jax import numpy as jnp
import jax


@jax.tree_util.register_pytree_node_class
class Identity:
    """
    Identity layer does nothing to the parameters. It is used to be base class for other layers.

    Parameters
    ----------
    dimensionality : int
        Dimensionality of the parameters.
    """

    random_initilization = False

    def __init__(self, dimensionality):
        self.in_features = dimensionality
        self.out_features = dimensionality

    @jax.jit
    def transform_params(self, params):
        return params

    def transform_preselect(self, preselect):
        return preselect

    def transform_group(self, group):
        return group

    def transform_sparsity(self, sparsity):
        return sparsity

    def tree_flatten(self):
        children = ()
        aux_data = {"dimensionality": self.in_features}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(**aux_data)


@jax.tree_util.register_pytree_node_class
class NonNegative(Identity):
    """
    NonNegative layer ensures that all parameters are non-negative.

    Parameters
    ----------
    dimensionality : int
        Dimensionality of the parameters.
    """

    @jax.jit
    def transform_params(self, params):
        return jnp.abs(params)


@jax.tree_util.register_pytree_node_class
class LinearConstraint(Identity):
    """
    LinearConstraint layer ensures that the parameters satisfy the linear constraint: ``<coef, params> = 1``.

    Parameters
    ----------
    dimensionality : int
        Dimensionality of the parameters.
    coef : float or array with shape (dimensionality,)
        Coefficients of the linear constraint ``<coef, params> = 1``.
        If ``coef`` is a float, then ``coef * ones(dimensionality)`` is used.
    """

    random_initilization = True

    def __init__(self, dimensionality, coef=None):
        if coef is None:
            coef = jnp.ones(dimensionality)
        if isinstance(coef, (int, float)):
            coef = jnp.ones(dimensionality) * coef
        assert coef.size == dimensionality
        self.in_features = dimensionality
        self.out_features = dimensionality
        self.coef = coef

    @jax.jit
    def transform_params(self, params):
        x = jnp.dot(params, self.coef)
        return params / jnp.where(x == 0.0, 1.0, x)

    def tree_flatten(self):
        children = (self.coef,)
        aux_data = {"dimensionality": self.in_features}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data["dimensionality"], *children)


@jax.tree_util.register_pytree_node_class
class SimplexConstraint(Identity):
    """
    SimplexConstraint layer ensures that the parameters satisfy the linear constraint: ``<coef, params> = 1`` and all parameters are non-negative.

    Parameters
    ----------
    dimensionality : int
        Dimensionality of the parameters.
    coef : float or array with shape (dimensionality,)
        Coefficients of the linear constraint ``<coef, params> = 1``.
        If ``coef`` is a float, then ``coef * ones(dimensionality)`` is used.
    """

    random_initilization = True

    def __init__(self, dimensionality, coef=None):
        if coef is None:
            coef = jnp.ones(dimensionality)
        assert coef.size == dimensionality
        self.in_features = dimensionality
        self.out_features = dimensionality
        self.coef = coef

    @jax.jit
    def transform_params(self, params):
        p = jnp.abs(params)
        x = jnp.dot(p, self.coef)
        return p / jnp.where(x == 0.0, 1.0, x)

    def tree_flatten(self):
        children = (self.coef,)
        aux_data = {"dimensionality": self.in_features}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data["dimensionality"], *children)


@jax.tree_util.register_pytree_node_class
class BoxConstraint(Identity):
    """
    BoxConstraint layer ensures that the parameters are in the box: ``lower <= params <= upper``.

    Parameters
    ----------
    dimensionality : int
        Dimensionality of the parameters.
    lower : float or array with shape (dimensionality,)
        Lower bound of the box, if ``lower`` is a float, then ``lower * ones(dimensionality)`` is used.
        ``lower`` must be non-positive.
    upper : float or array with shape (dimensionality,)
        Upper bound of the box, if ``upper`` is a float, then ``upper * ones(dimensionality)`` is used.
        ``upper`` must be non-negative.
    """

    def __init__(self, dimensionality, lower, upper):
        self.in_features = dimensionality
        self.out_features = dimensionality
        self.lower = jnp.zeros(dimensionality) + lower
        self.upper = jnp.zeros(dimensionality) + upper

    @jax.jit
    def transform_params(self, params):
        return jnp.clip(params, self.lower, self.upper)

    def tree_flatten(self):
        children = (self.lower, self.upper)
        aux_data = {"dimensionality": self.in_features}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data["dimensionality"], *children)


@jax.tree_util.register_pytree_node_class
class OffsetSparse(Identity):
    """
    OffsetSparse layer ensures that the sparse constraint of sparse solvers changes from ``||params||_0 = s`` to ``||params - offset||_0 = s``. In other words, the layer ensures that the parameters corresponding to the non-selected features are equal to ``offset`` rather than zero.

    Parameters
    ----------
    dimensionality : int
        Dimensionality of the parameters.
    offset : float or array with shape (dimensionality,)
        Offset of the sparse constraint.
    """

    def __init__(self, dimensionality, offset):
        self.in_features = dimensionality
        self.out_features = dimensionality
        self.offset = jnp.zeros(dimensionality) + offset

    @jax.jit
    def transform_params(self, params):
        return params + self.offset

    def tree_flatten(self):
        children = (self.offset,)
        aux_data = {"dimensionality": self.out_features}
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(aux_data["dimensionality"], *children)


if __name__ == "__main__":
    params = jnp.array([1, -1])
    layer = LinearConstraint(2)
    print(layer.transform_params(params))
