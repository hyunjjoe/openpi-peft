import dataclasses

import flax.nnx as nnx


@dataclasses.dataclass(frozen=True)
class TopKLayerFreezeConfig:
    """Configuration for top-k layer fine-tuning implemented as a freeze scheme.

    This does *not* add any new parameters. Instead, it specifies which transformer
    blocks of the (bridged) Gemma model should remain trainable.

    The resulting filter follows the convention used in `TrainConfig`:
    it returns **True** for parameters that should be *frozen*.
    """

    # Total number of transformer blocks in the Gemma stack.
    total_layers: int
    # Number of top (closest to output) layers to keep trainable.
    k_unfrozen: int

    def make_freeze_filter(self) -> nnx.filterlib.Filter:
        """Return an nnx filter that freezes all but the top-k Gemma layers.

        We identify layers via the `"block_<idx>"` segment in the path under the
        bridged LLM module (which contains `"llm"` in its path). This matches the
        layout used by `gemma_topk.TopKModule`.

        - If `k_unfrozen <= 0`, all LLM layers are frozen.
        - If `k_unfrozen >= total_layers`, no layer-level freezing is applied.
        """
        # Edge cases.
        if self.k_unfrozen <= 0:
            # Freeze all LLM parameters.
            def freeze_all_llm(path: nnx.filterlib.PathParts, x) -> bool:  # noqa: ANN001
                return any(str(p) == "llm" for p in path)

            return freeze_all_llm

        if self.k_unfrozen >= self.total_layers:
            # No per-layer freezing needed.
            return nnx.Nothing

        cutoff = self.total_layers - self.k_unfrozen

        def _freeze_early_layers(path: nnx.filterlib.PathParts, x) -> bool:  # noqa: ANN001
            # Only operate on the LLM subtree.
            parts = [str(p) for p in path]
            if "llm" not in parts:
                return False

            # Find "block_<idx>" in the path, if present.
            for p in parts:
                if p.startswith("block_"):
                    try:
                        layer_idx = int(p.split("_", maxsplit=1)[1])
                    except ValueError:
                        return False
                    # Freeze all layers strictly before the cutoff; keep the top-k trainable.
                    return layer_idx < cutoff

            return False

        return _freeze_early_layers
