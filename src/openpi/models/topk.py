import dataclasses

import flax.nnx as nnx


@dataclasses.dataclass(frozen=True)
class TopKLayerFreezeConfig:
    """Configuration for top-k layer fine-tuning implemented as a freeze scheme.

    This does not add any new parameters. Instead, it specifies which transformer
    blocks of the (bridged) Gemma model should remain trainable.

    The resulting filter follows the convention used in `TrainConfig`:
    it returns True for parameters that should be frozen.
    """

    # Total number of transformer blocks in the Gemma stack.
    total_layers: int
    # Number of top (closest to output) layers to keep trainable.
    k_unfrozen: int
    # Whether to apply top-k freezing to the PaliGemma expert (no "_1" suffix).
    include_pali: bool = True
    # Whether to apply top-k freezing to the action expert (names with "_1" suffix).
    include_action: bool = True

    def make_freeze_filter(self) -> nnx.filterlib.Filter:
        """Return an nnx filter that freezes all but the top-k Gemma layers.

        We identify layers via the "block_<idx>" segment in the path under the
        bridged LLM module (which contains "llm" in its path). This matches the
        layout used by gemma_topk.TopKModule.

        Within the selected experts (see include_pali / include_action), this
        implements:
          - For blocks with idx < cutoff (= total_layers - k_unfrozen): freeze all
            parameters in those blocks.
          - For blocks with idx >= cutoff: freeze everything *except* the attention
            output projections (`attn_vec_einsum`) which remain unfrozen.

        - If k_unfrozen <= 0, all LLM layers are frozen.
        - If k_unfrozen >= total_layers, no layer-level freezing is applied.
        """
        if self.k_unfrozen <= 0:
            # Freeze all LLM parameters (both experts).
            def freeze_all_llm(path: nnx.filterlib.PathParts, x) -> bool:  # noqa: ANN001
                return any(str(p) == "llm" for p in path)

            return freeze_all_llm

        if self.k_unfrozen >= self.total_layers:
            return nnx.Nothing

        cutoff = self.total_layers - self.k_unfrozen

        def _freeze_early_layers(path: nnx.filterlib.PathParts, x) -> bool:  # noqa: ANN001
            # Only operate on the LLM subtree.
            parts = [str(p) for p in path]
            if "llm" not in parts:
                return False

            # Determine which expert this parameter belongs to.
            is_action = any(p.endswith("_1") for p in parts)
            is_pali = not is_action

            if is_action and not self.include_action:
                return True
            if is_pali and not self.include_pali:
                return True

            subject = (self.include_pali and is_pali) or (self.include_action and is_action)
            if not subject:
                return False

            for p in parts:
                if p.startswith("block_"):
                    try:
                        layer_idx = int(p.split("_", maxsplit=1)[1])
                    except ValueError:
                        return True  # conservative: freeze on malformed index

                    # For early blocks, freeze everything.
                    if layer_idx < cutoff:
                        return True

                    # For the last k blocks, only leave the attention output projections
                    # (`attn_vec_einsum`) unfrozen; freeze everything else.
                    if "attn_vec_einsum" in parts:
                        return False
                    return True

            # LLM params outside of any block_* (e.g., embedder, final norms) are frozen.
            return True

        return _freeze_early_layers
