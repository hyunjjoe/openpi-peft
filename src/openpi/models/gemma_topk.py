"""Gemma module variant with per-layer blocks, to support top-k layer fine-tuning.

This module reuses the existing Gemma building blocks (Config, Attention, Block, etc.)
but exposes each transformer block as its own named submodule:

    ['PaliGemma']['llm']['block_0'][...]
    ['PaliGemma']['llm']['block_1'][...]
    ...

This makes it possible to freeze or unfreeze specific layers using an NNX filter that
matches on the block name (e.g., 'block_0', 'block_1', ...), without changing the
behavior of the standard Gemma module used by other PEFT methods.
"""

from collections.abc import Sequence

import einops
import flax.linen as nn
import jax
import jax.numpy as jnp

from openpi.models.gemma import (  # type: ignore[attr-defined]
    Config,
    KVCache,
    PALIGEMMA_VOCAB_SIZE,
    Block,
    Embedder,
    RMSNorm,
)
import openpi.shared.array_typing as at


@at.typecheck
class TopKModule(nn.Module):
    """Gemma transformer with per-layer blocks.

    The main differences from `openpi.models.gemma.Module` are:
      - We *do not* use `nn.scan` with a params axis over depth.
      - Instead, we create one `Block` submodule per layer, named `block_0`, ..., `block_{depth-1}`.
      - This makes the paths of the parameters include the block index, which can be used
        by NNX filters to implement top-k layer freezing.

    The forward semantics (including KV caching) match the original module.
    KV caching is fully supported for efficient autoregressive decoding.
    """

    configs: Sequence[Config]
    embed_dtype: str

    dropout: float = 0.0
    dropout_bdims: tuple[int, ...] = ()
    adarms: bool = False

    def setup(self):
        # all experts must have the same depth
        assert all(config.depth == self.configs[0].depth for config in self.configs)

        self.embedder = Embedder(
            vocab_size=PALIGEMMA_VOCAB_SIZE,
            embed_dim=self.configs[0].width,  # embedder for first expert only
            name="embedder",
        )

        # Create one Block submodule per layer, each with its own name (block_0, block_1, ...).
        depth = self.configs[0].depth
        self.blocks = [
            Block(
                configs=tuple(self.configs),
                dropout=self.dropout,
                dropout_bdims=self.dropout_bdims,
                name=f"block_{layer_idx}",
            )
            for layer_idx in range(depth)
        ]
        self.final_norms = [RMSNorm(name=f"final_norm_{i}" if i > 0 else "final_norm") for i in range(len(self.configs))]

    @at.typecheck
    def embed(self, tokens: at.Int[at.Array, "b t"]) -> at.Float[at.Array, "b t d"]:
        return self.embedder.encode(tokens).astype(self.embed_dtype)

    @at.typecheck
    def __call__(
        self,
        # list of token arrays, one for each expert, or None if that expert should not be run
        embedded: Sequence[at.Float[at.Array, "b _t _d"] | None],
        positions: at.Int[at.Array, "b t"],
        mask: at.Bool[at.Array, "b t s"],
        adarms_cond: Sequence[at.Float[at.Array, "b _d"] | None] | None = None,
        *,
        kv_cache: KVCache | None = None,
        deterministic: bool = True, 
    ) -> tuple[Sequence[at.Float[at.Array, "b _t _d"] | None], KVCache]:
        """Forward pass with KV caching support.

        When kv_cache is provided, uses cached key-value pairs from previous forward passes
        to speed up autoregressive generation. The cache format matches the standard Gemma module:
        a tuple of (cache_k, cache_v) where each has shape [layers, batch, seq, heads, dim].
        """
        embedded = jax.tree.map(lambda e: e.astype(self.embed_dtype) if e is not None else None, embedded)
        mask4d = jnp.asarray(mask)[:, None, :, :]
        if adarms_cond is None:
            adarms_cond = [None] * len(self.configs)

        xs = embedded
        depth = len(self.blocks)
        
        # Handle KV cache: split by layer if provided
        if kv_cache is not None:
            # kv_cache is (cache_k, cache_v) -> each has shape [layers, batch, seq, heads, dim]
            cache_k, cache_v = kv_cache
            layer_caches = [(cache_k[i], cache_v[i]) for i in range(depth)]
        else:
            layer_caches = [None] * depth
        
        updated_cache_k = []
        updated_cache_v = []
        
        for layer_idx, block in enumerate(self.blocks):
            layer_cache = layer_caches[layer_idx]
            xs, new_layer_cache = block(
                xs, 
                kv_cache=layer_cache, 
                positions=positions, 
                attn_mask=mask4d, 
                adarms_cond=adarms_cond, 
                deterministic=deterministic
            )
            
            # Collect updated cache from this layer (even if creating new cache)
            if new_layer_cache is not None:
                new_k, new_v = new_layer_cache
                updated_cache_k.append(new_k)
                updated_cache_v.append(new_v)

        assert all(e.dtype == jnp.dtype(self.embed_dtype) for e in xs if e is not None)
        
        # Reconstruct cache in the same format as standard Gemma
        new_kv_cache = (jnp.stack(updated_cache_k), jnp.stack(updated_cache_v))

        outputs = [
            f(e, a)[0] if e is not None else e for f, e, a in zip(self.final_norms, xs, adarms_cond, strict=True)
        ]
        return outputs, new_kv_cache

    def init(self, use_adarms: Sequence[bool]):
        """Convenience method for initializing all parameters.

        Mirrors `gemma.Module.init` so that `nnx_bridge.ToNNX.lazy_init(..., method="init", ...)`
        works consistently for both the standard and top-k Gemma modules.
        """
        self.embed(jnp.zeros((1, 1), dtype=jnp.int32))
        # Run a dummy forward pass to initialize all block parameters.
        self(
            [jnp.zeros((1, 1, c.width)) for c in self.configs],
            jnp.zeros((1, len(self.configs)), dtype=jnp.int32),
            jnp.zeros((1, len(self.configs), len(self.configs)), dtype=bool),
            adarms_cond=[
                jnp.zeros((1, c.width)) if u else None for u, c in zip(use_adarms, self.configs, strict=True)
            ],
        )