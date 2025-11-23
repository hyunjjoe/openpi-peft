import flax.linen as nn
import jax
import jax.numpy as jnp

import openpi.models.adapters as adapters


def test_adapter_layer_params_shape():
    cfg = adapters.AdapterConfig(reduction_factor=8)
    layer = adapters.AdapterLayer(hidden_dim=32, config=cfg)
    key = jax.random.key(0)
    x = jax.random.normal(key, (2, 32))
    params = layer.init(key, x, deterministic=True)
    assert params["params"]["down"]["kernel"].shape == (32, 4)
    assert params["params"]["up"]["kernel"].shape == (4, 32)


def test_adapter_layer_identity_with_zero_init():
    cfg = adapters.AdapterConfig(reduction_factor=4, kernel_init=nn.initializers.zeros)
    layer = adapters.AdapterLayer(hidden_dim=16, config=cfg)
    key = jax.random.key(0)
    x = jax.random.normal(key, (4, 16))
    params = layer.init(key, x, deterministic=True)
    y = layer.apply(params, x, deterministic=True)
    assert jnp.allclose(x, y)

