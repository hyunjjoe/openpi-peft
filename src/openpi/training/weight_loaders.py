import dataclasses
import logging
import re
from typing import Protocol, runtime_checkable

import flax.traverse_util
import numpy as np

import openpi.models.model as _model
import openpi.shared.array_typing as at
import openpi.shared.download as download

logger = logging.getLogger(__name__)


@runtime_checkable
class WeightLoader(Protocol):
    def load(self, params: at.Params) -> at.Params:
        """Loads the model weights.

        Args:
            params: Parameters of the model. This is a nested structure of array-like objects that
                represent the model's parameters.

        Returns:
            Loaded parameters. The structure must be identical to `params`. If returning a subset of
            the parameters the loader must merge the loaded parameters with `params`.
        """


@dataclasses.dataclass(frozen=True)
class NoOpWeightLoader(WeightLoader):
    def load(self, params: at.Params) -> at.Params:
        return params


@dataclasses.dataclass(frozen=True)
class CheckpointWeightLoader(WeightLoader):
    """Loads an entire set of weights from a checkpoint.

    Compatible with:
      trained checkpoints:
        example: "./checkpoints/<config>/<exp>/<step>/params"
      released checkpoints:
        example: "gs://openpi-assets/checkpoints/<model>/params"
    """

    params_path: str

    def load(self, params: at.Params) -> at.Params:
        # We are loading np.ndarray and relying on the training code to properly convert and shard the params.
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)
        # Add all missing LoRA weights.
        return _merge_params(loaded_params, params, missing_regex=".*(lora|adapter|kv_prefix).*")


@dataclasses.dataclass(frozen=True)
class TopKGemmaCheckpointWeightLoader(WeightLoader):
    """Loads Gemma weights from a standard checkpoint into a top-k Gemma layout.

    The standard Gemma module stores transformer block parameters stacked along a
    leading depth axis under:

        .../PaliGemma/llm/layers/...

    The top-k Gemma module (`gemma_topk.TopKModule`) instead exposes one submodule
    per block:

        .../PaliGemma/llm/block_0/...
        .../PaliGemma/llm/block_1/...
        ...

    This loader maps from the former to the latter by slicing along the depth axis.
    All non-LLM parameters (e.g., image tower, projections) are copied directly
    when present in the checkpoint; any missing parameters fall back to the
    reference `params` (e.g., new adapters or other PEFT weights).
    """

    params_path: str
    # Total number of transformer blocks in the Gemma stack.
    total_layers: int

    def load(self, params: at.Params) -> at.Params:
        loaded_params = _model.restore_params(download.maybe_download(self.params_path), restore_type=np.ndarray)

        flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
        flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

        result: dict[tuple[str, ...], at.Array | at.Params] = {}

        for k_ref, v_ref in flat_ref.items():
            # k_ref is a tuple of path components.
            path_str = "/".join(str(p) for p in k_ref)

            # If this parameter is NOT under PaliGemma/llm/block_<i>, try to copy it
            # directly from the checkpoint when present; otherwise keep the reference.
            if "PaliGemma/llm/block_" not in path_str:
                if k_ref in flat_loaded:
                    v_loaded = flat_loaded[k_ref]
                    result[k_ref] = v_loaded.astype(v_ref.dtype) if v_loaded.dtype != v_ref.dtype else v_loaded
                else:
                    result[k_ref] = v_ref
                continue

            # For top-k Gemma blocks, map:
            #   .../PaliGemma/llm/block_i/<rest>  ->  .../PaliGemma/llm/layers/<rest>[i]
            parts = [str(p) for p in k_ref]
            try:
                block_idx_pos = next(i for i, p in enumerate(parts) if p.startswith("block_"))
            except StopIteration:
                # Should not happen given the string check above; fall back to reference.
                result[k_ref] = v_ref
                continue

            block_token = parts[block_idx_pos]
            try:
                layer_idx = int(block_token.split("_", maxsplit=1)[1])
            except ValueError:
                result[k_ref] = v_ref
                continue

            if not (0 <= layer_idx < self.total_layers):
                result[k_ref] = v_ref
                continue

            # Build the corresponding "layers/..." key in the checkpoint.
            old_parts = parts.copy()
            old_parts[block_idx_pos] = "layers"
            k_old = tuple(old_parts)

            if k_old not in flat_loaded:
                # No matching stacked parameter in the checkpoint; keep the reference.
                result[k_ref] = v_ref
                continue

            v_old = flat_loaded[k_old]
            if v_old.ndim == 0 or v_old.shape[0] != self.total_layers:
                # Unexpected shape; be conservative and keep the reference.
                result[k_ref] = v_ref
                continue

            # Slice along the depth axis for this block.
            v_block = v_old[layer_idx]
            result[k_ref] = v_block.astype(v_ref.dtype) if v_block.dtype != v_ref.dtype else v_block

        return flax.traverse_util.unflatten_dict(result, sep="/")


@dataclasses.dataclass(frozen=True)
class PaliGemmaWeightLoader(WeightLoader):
    """Loads weights from the official PaliGemma checkpoint.

    This will overwrite existing weights with similar names while keeping all extra weights intact.
    This allows us to support the action expert which is used by the Pi0 model.
    """

    def load(self, params: at.Params) -> at.Params:
        path = download.maybe_download(
            "gs://vertex-model-garden-paligemma-us/paligemma/pt_224.npz", gs={"token": "anon"}
        )
        with path.open("rb") as f:
            flat_params = dict(np.load(f, allow_pickle=False))
        loaded_params = {"PaliGemma": flax.traverse_util.unflatten_dict(flat_params, sep="/")["params"]}
        # Add all missing weights.
        return _merge_params(loaded_params, params, missing_regex=".*")


def _merge_params(loaded_params: at.Params, params: at.Params, *, missing_regex: str) -> at.Params:
    """Merges the loaded parameters with the reference parameters.

    Args:
        loaded_params: The parameters to merge.
        params: The reference parameters.
        missing_regex: A regex pattern for all missing keys that should be merged from the reference parameters.

    Returns:
        A new dictionary with the merged parameters.
    """
    flat_ref = flax.traverse_util.flatten_dict(params, sep="/")
    flat_loaded = flax.traverse_util.flatten_dict(loaded_params, sep="/")

    # First, take all weights that are a subset of the reference weights.
    result = {}
    for k, v in flat_loaded.items():
        if k in flat_ref:
            result[k] = v.astype(flat_ref[k].dtype) if v.dtype != flat_ref[k].dtype else v

    flat_loaded.clear()

    # Then, merge any missing weights as defined by the missing regex.
    pattern = re.compile(missing_regex)
    for k in {k for k in flat_ref if pattern.fullmatch(k)}:
        if k not in result:
            result[k] = flat_ref[k]

    return flax.traverse_util.unflatten_dict(result, sep="/")
