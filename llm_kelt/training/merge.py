"""Layer-by-layer LoRA merge for memory-constrained environments.

For large BNB 4-bit models (e.g., 72B), the standard merge_and_unload() requires
dequantizing the entire model (~144GB for 72B), which may exceed available RAM.

This module provides a memory-efficient alternative that processes one layer at a time,
with peak memory of ~2-3GB per layer instead of the full model.

Upstream issues tracking BNB merge/save limitations:
- https://github.com/huggingface/transformers/issues/23904
- https://github.com/huggingface/peft/issues/2321
- https://github.com/huggingface/peft/issues/2501

TODO: Remove this workaround once upstream fix is available.
"""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from appinfra.log import Logger


def _load_lora_map(adapter_path: Path) -> tuple[dict[str, tuple[Any, Any]], float]:
    """Load LoRA adapter weights and build base_key -> (lora_A, lora_B) mapping.

    Returns:
        Tuple of (lora_map, scaling_factor).
    """
    from safetensors.torch import load_file

    with open(adapter_path / "adapter_config.json") as f:
        config = json.load(f)

    lora_alpha = config.get("lora_alpha", 16)
    lora_r = config.get("r", 8)
    scaling = lora_alpha / lora_r

    adapter_weights = load_file(adapter_path / "adapter_model.safetensors")

    lora_map: dict[str, tuple[Any, Any]] = {}
    for key in adapter_weights:
        if ".lora_A." in key:
            # Key format: base_model.model.layers.0.self_attn.q_proj.lora_A.weight
            base_key = (
                key.replace("base_model.model.", "")
                .replace(".lora_A.weight", ".weight")
                .replace(".lora_A.default.weight", ".weight")
            )
            b_key = key.replace(".lora_A.", ".lora_B.")
            if b_key in adapter_weights:
                lora_map[base_key] = (adapter_weights[key], adapter_weights[b_key])

    return lora_map, scaling


def _load_weight_index(model_path: Path) -> tuple[dict[str, str], dict[str, list[str]]]:
    """Load model weight index and group weights by source file.

    Returns:
        Tuple of (weight_map, file_to_weights).
    """
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.exists():
        raise ValueError(f"Expected sharded model with index at {index_path}")

    with open(index_path) as f:
        base_index = json.load(f)

    weight_map = base_index.get("weight_map", {})

    file_to_weights: dict[str, list[str]] = {}
    for weight_name, filename in weight_map.items():
        file_to_weights.setdefault(filename, []).append(weight_name)

    return weight_map, file_to_weights


def _is_quant_metadata_key(weight_name: str) -> bool:
    """Check if weight name is BNB quantization metadata (not actual weights)."""
    return any(
        x in weight_name
        for x in [".weight_format", ".quant_state.", ".absmax", ".quant_map", ".nested"]
    )


def _is_quantized_weight(weight_name: str, weight_map: dict[str, str]) -> bool:
    """Check if weight has BNB quantization metadata."""
    return f"{weight_name}.absmax" in weight_map


def _load_quant_tensor(
    name: str,
    weight_name: str,
    file_handle: Any,
    weight_map: dict[str, str],
    model_path: Path,
) -> Any:
    """Load a quantization-related tensor from the appropriate safetensor file."""
    from safetensors import safe_open

    filename = weight_map.get(name)
    if filename is None:
        return None
    if filename == weight_map[weight_name]:
        return file_handle.get_tensor(name)
    with safe_open(model_path / filename, framework="pt", device="cpu") as f:
        return f.get_tensor(name)


def _build_quant_state(
    absmax: Any,
    quant_map: Any,
    nested_absmax: Any,
    nested_quant_map: Any,
    metadata: dict[str, Any],
) -> Any:
    """Build BNB QuantState from component tensors and JSON metadata."""
    import torch
    from bitsandbytes.functional import QuantState

    # Build nested state for double quantization if present
    state2 = None
    offset = None
    if nested_absmax is not None:
        state2 = QuantState(
            absmax=nested_absmax,
            blocksize=metadata["nested_blocksize"],
            code=nested_quant_map,
            dtype=getattr(torch, metadata["nested_dtype"]),
        )
        offset = torch.tensor(metadata["nested_offset"])

    return QuantState(
        absmax=absmax,
        shape=torch.Size(metadata["shape"]),
        code=quant_map,
        blocksize=metadata["blocksize"],
        quant_type=metadata["quant_type"],
        dtype=getattr(torch, metadata["dtype"]),
        offset=offset,
        state2=state2,
    )


def _dequantize_bnb_weight(
    weight_name: str, file_handle: Any, weight_map: dict[str, str], model_path: Path
) -> Any:
    """Dequantize a BNB 4-bit weight using associated quant state."""
    from bitsandbytes.functional import dequantize_4bit

    quant_weight = file_handle.get_tensor(weight_name)

    def load(suffix: str) -> Any:
        return _load_quant_tensor(
            f"{weight_name}.{suffix}", weight_name, file_handle, weight_map, model_path
        )

    # Load and parse quant_state JSON metadata
    qs_tensor = load("quant_state.bitsandbytes__nf4")
    metadata = json.loads(bytes(qs_tensor.numpy()).decode("utf-8"))

    absmax = load("absmax")
    quant_map = load("quant_map")
    nested_absmax = load("nested_absmax")
    nested_quant_map = load("nested_quant_map")

    quant_state = _build_quant_state(absmax, quant_map, nested_absmax, nested_quant_map, metadata)
    return dequantize_4bit(quant_weight, quant_state)


def _apply_lora(tensor: Any, lora_a: Any, lora_b: Any, scaling: float, dtype: Any) -> Any:
    """Apply LoRA delta to tensor: W' = W + (B @ A) * scaling."""
    delta = (lora_b.to(dtype) @ lora_a.to(dtype)) * scaling
    # Reshape tensor if needed (dequantized BNB weights may be flat or [1, N])
    if tensor.shape != delta.shape:
        tensor = tensor.view(delta.shape)
    tensor = tensor.to(dtype)
    return tensor + delta


class ShardWriter:
    """Accumulates tensors and writes them to sharded safetensor files."""

    def __init__(self, output_path: Path, max_shard_size: int = 5 * 1024**3):
        from safetensors.torch import save_file

        self._save_file = save_file
        self._output_path = output_path
        self._max_shard_size = max_shard_size
        self._weight_map: dict[str, str] = {}
        self._total_size = 0
        self._shard_idx = 0
        self._current_shard: dict[str, Any] = {}
        self._current_shard_size = 0

    def add(self, name: str, tensor: Any) -> None:
        """Add tensor to current shard, flushing if needed."""
        tensor = tensor.contiguous()
        tensor_size = tensor.numel() * tensor.element_size()

        if self._current_shard_size + tensor_size > self._max_shard_size:
            self.flush()

        self._current_shard[name] = tensor
        self._current_shard_size += tensor_size

    def flush(self) -> int:
        """Write current shard to disk. Returns number of keys written."""
        if not self._current_shard:
            return 0

        shard_name = f"model-{self._shard_idx:05d}-of-XXXXX.safetensors"
        self._save_file(self._current_shard, self._output_path / shard_name)

        for k in self._current_shard:
            self._weight_map[k] = shard_name

        count = len(self._current_shard)
        self._total_size += self._current_shard_size
        self._shard_idx += 1
        self._current_shard = {}
        self._current_shard_size = 0
        gc.collect()

        return count

    def finalize(self) -> tuple[int, int]:
        """Flush remaining data, rename shards, write index. Returns (shard_count, total_size)."""
        self.flush()

        # Rename shards with correct total count
        final_count = self._shard_idx
        for i in range(final_count):
            old_name = f"model-{i:05d}-of-XXXXX.safetensors"
            new_name = f"model-{i + 1:05d}-of-{final_count:05d}.safetensors"
            (self._output_path / old_name).rename(self._output_path / new_name)
            for k, v in self._weight_map.items():
                if v == old_name:
                    self._weight_map[k] = new_name

        # Write index
        index = {"metadata": {"total_size": self._total_size}, "weight_map": self._weight_map}
        with open(self._output_path / "model.safetensors.index.json", "w") as f:
            json.dump(index, f, indent=2)

        return final_count, self._total_size


def _process_weight_file(
    model_path: Path,
    filepath: Path,
    weight_names: list[str],
    weight_map: dict[str, str],
    lora_map: dict[str, tuple[Any, Any]],
    scaling: float,
    dtype: Any,
    writer: ShardWriter,
) -> None:
    """Process a single safetensor file, applying LoRA where applicable."""
    from safetensors import safe_open

    with safe_open(filepath, framework="pt", device="cpu") as f:
        for weight_name in weight_names:
            if _is_quant_metadata_key(weight_name):
                continue

            # Dequantize BNB weights before processing
            if _is_quantized_weight(weight_name, weight_map):
                tensor = _dequantize_bnb_weight(weight_name, f, weight_map, model_path)
            else:
                tensor = f.get_tensor(weight_name)
            if weight_name in lora_map:
                lora_a, lora_b = lora_map[weight_name]
                tensor = _apply_lora(tensor, lora_a, lora_b, scaling, dtype)
            else:
                tensor = tensor.to(dtype)
            writer.add(weight_name, tensor)

    gc.collect()


def merge_layerwise(
    lg: Logger,
    model_path: Path,
    adapter_path: Path,
    output_path: Path,
    dtype: Any,
) -> None:
    """Merge LoRA adapter into base model layer-by-layer.

    Processes one safetensor file at a time with peak memory of ~2-3GB per layer.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    lora_map, scaling = _load_lora_map(adapter_path)
    lg.info("loaded LoRA adapter", extra={"layers": len(lora_map), "scaling": scaling})

    weight_map, file_to_weights = _load_weight_index(model_path)
    writer = ShardWriter(output_path)

    sorted_files = sorted(file_to_weights.keys())
    for file_idx, filename in enumerate(sorted_files, 1):
        lg.info("processing shard", extra={"progress": f"{file_idx}/{len(sorted_files)}"})
        _process_weight_file(
            model_path,
            model_path / filename,
            file_to_weights[filename],
            weight_map,
            lora_map,
            scaling,
            dtype,
            writer,
        )

    shard_count, total_size = writer.finalize()
    lg.info(
        "layerwise merge complete",
        extra={"shards": shard_count, "total_size_gb": round(total_size / 1024**3, 2)},
    )


def _quantize_tensor_bnb(tensor: Any) -> tuple[Any, Any]:
    """Quantize a tensor to BNB 4-bit NF4 format with double quantization."""
    import torch
    from bitsandbytes.functional import quantize_4bit

    tensor = tensor.contiguous().cuda()
    quantized, quant_state = quantize_4bit(
        tensor, quant_type="nf4", blocksize=64, compress_statistics=True
    )
    # Move back to CPU immediately to free GPU memory
    quantized = quantized.cpu()
    quant_state.absmax = quant_state.absmax.cpu()
    if quant_state.code is not None:
        quant_state.code = quant_state.code.cpu()
    if quant_state.state2 is not None:
        quant_state.state2.absmax = quant_state.state2.absmax.cpu()
        if quant_state.state2.code is not None:
            quant_state.state2.code = quant_state.state2.code.cpu()
    torch.cuda.empty_cache()
    return quantized, quant_state


def _get_nested_offset(qs: Any) -> float | None:
    """Extract nested offset value from quant state."""
    import torch

    if qs.state2 is None:
        return None
    offset_val = qs.offset
    if isinstance(offset_val, torch.Tensor):
        offset_val = offset_val.item()
    return float(offset_val) if offset_val is not None else 0.0


def _build_bnb_metadata(qs: Any, nested_offset: float | None) -> dict:
    """Build BNB quant state metadata dict."""
    return {
        "blocksize": 64,
        "quant_type": "nf4",
        "dtype": str(qs.dtype).split(".")[-1],
        "shape": list(qs.shape),
        "nested_blocksize": 256 if qs.state2 else None,
        "nested_dtype": str(qs.state2.dtype).split(".")[-1] if qs.state2 else None,
        "nested_offset": nested_offset,
    }


def _add_bnb_weights(writer: ShardWriter, name: str, quantized: Any, qs: Any) -> None:
    """Add BNB quantized weight and its quant state tensors to writer."""
    import torch

    writer.add(name, quantized)
    writer.add(f"{name}.absmax", qs.absmax)
    writer.add(f"{name}.quant_map", qs.code)

    nested_offset = _get_nested_offset(qs)
    metadata = _build_bnb_metadata(qs, nested_offset)
    packed = torch.tensor([ord(c) for c in json.dumps(metadata)], dtype=torch.uint8)
    writer.add(f"{name}.quant_state.bitsandbytes__nf4", packed)

    if qs.state2 is not None:
        writer.add(f"{name}.nested_absmax", qs.state2.absmax)
        writer.add(f"{name}.nested_quant_map", qs.state2.code)


def _should_quantize(name: str) -> bool:
    """Check if a weight should be quantized (skip embeddings, norms, biases, etc.)."""
    skip_patterns = ["embed", "norm", "lm_head", "rotary", ".bias"]
    return not any(p in name.lower() for p in skip_patterns)


def quantize_layerwise(lg: Logger, model_path: Path, output_path: Path) -> None:
    """Quantize a fp16 model to BNB 4-bit layer-by-layer."""
    output_path.mkdir(parents=True, exist_ok=True)

    weight_map, file_to_weights = _load_weight_index(model_path)
    writer = ShardWriter(output_path)

    sorted_files = sorted(file_to_weights.keys())
    for file_idx, filename in enumerate(sorted_files, 1):
        lg.info("quantizing shard", extra={"progress": f"{file_idx}/{len(sorted_files)}"})
        _quantize_weight_file(model_path / filename, file_to_weights[filename], writer)

    shard_count, total_size = writer.finalize()
    lg.info(
        "layerwise quantization complete",
        extra={"shards": shard_count, "total_size_gb": round(total_size / 1024**3, 2)},
    )


def _quantize_weight_file(filepath: Path, weight_names: list[str], writer: ShardWriter) -> None:
    """Quantize weights from a single safetensor file."""
    from safetensors import safe_open

    with safe_open(filepath, framework="pt", device="cpu") as f:
        for weight_name in weight_names:
            tensor = f.get_tensor(weight_name)
            if _should_quantize(weight_name):
                quantized, quant_state = _quantize_tensor_bnb(tensor)
                _add_bnb_weights(writer, weight_name, quantized, quant_state)
            else:
                writer.add(weight_name, tensor)
    gc.collect()
