import collections.abc as cabc

import flax.linen as nn
import flax.struct as struct
import jax.numpy as jnp


@struct.dataclass
class AdapterConfig:
    reduction_factor: int = 16
    non_linearity: cabc.Callable[[jnp.ndarray], jnp.ndarray] = nn.gelu
    dropout_rate: float = 0.0
    scaling: float = 1.0
    kernel_init: nn.initializers.Initializer = nn.initializers.xavier_uniform()
    use_bias: bool = False


class AdapterLayer(nn.Module):
    hidden_dim: int
    config: AdapterConfig

    @nn.compact
    def __call__(self, x: jnp.ndarray, *, deterministic: bool) -> jnp.ndarray:
        dtype = x.dtype
        bottleneck = max(1, self.hidden_dim // max(1, self.config.reduction_factor))
        down = nn.Dense(
            bottleneck,
            use_bias=self.config.use_bias,
            kernel_init=self.config.kernel_init,
            dtype=dtype,
            name="down",
        )
        up = nn.Dense(
            self.hidden_dim,
            use_bias=self.config.use_bias,
            kernel_init=self.config.kernel_init,
            dtype=dtype,
            name="up",
        )
        z = down(x)
        z = self.config.non_linearity(z)
        if self.config.dropout_rate:
            z = nn.Dropout(rate=self.config.dropout_rate)(z, deterministic=deterministic)
        z = up(z)
        return x + z.astype(dtype) * self.config.scaling

