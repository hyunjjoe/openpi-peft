import flax.linen as nn
import flax.struct as struct
import jax.numpy as jnp


@struct.dataclass
class PrefixConfig:
    num_prefix_tokens: int = 16
    init_std: float = 0.02
    dropout_rate: float = 0.0


class KVPrefix(nn.Module):
    num_kv_heads: int
    head_dim: int
    config: PrefixConfig

    @nn.compact
    def __call__(
        self,
        batch_size: int,
        *,
        deterministic: bool,
        dtype=jnp.bfloat16,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:

        p = self.config.num_prefix_tokens
        k = self.param(
            "k",
            nn.initializers.normal(self.config.init_std),
            (p, self.num_kv_heads, self.head_dim),
        ).astype(dtype)
        v = self.param(
            "v",
            nn.initializers.normal(self.config.init_std),
            (p, self.num_kv_heads, self.head_dim),
        ).astype(dtype)

        k = jnp.broadcast_to(k[None, ...], (batch_size, p, self.num_kv_heads, self.head_dim))
        v = jnp.broadcast_to(v[None, ...], (batch_size, p, self.num_kv_heads, self.head_dim))

        if self.config.dropout_rate:
            drop = nn.Dropout(rate=self.config.dropout_rate)
            k = drop(k, deterministic=deterministic)
            v = drop(v, deterministic=deterministic)

        return k, v
