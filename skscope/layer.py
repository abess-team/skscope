import numpy as np
from jax import numpy as jnp
import jax

"""
must transform_params(0) == 0 !!

p_optim -1, 3
L1 non-negative   1, 3
L2 sum_up_to_one  0.25, 0.75
p_user 0.25, 0.75
loss 
"""


@jax.tree_util.register_pytree_node_class
class Identity:
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
    @jax.jit
    def transform_params(self, params):
        return jnp.abs(params)


@jax.tree_util.register_pytree_node_class
class LinearConstraint(Identity):
    """
    constraint: coef * params = 1
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
    constraint: coef * params = 1 ans params >= 0
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
    def __init__(self, dimensionality, lower, upper):
        # TODO check lower <= 0 <= upper
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
