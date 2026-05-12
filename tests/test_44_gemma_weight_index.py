from __future__ import annotations

import json
import struct
import tempfile
from pathlib import Path

from gemma_runtime.weights import GemmaWeightIndex, SafetensorsShard


def write_safetensors(path: Path, tensors: dict[str, tuple[str, list[int], bytes]]) -> None:
    header = {}
    offset = 0
    data_parts = []
    for name, (dtype, shape, data) in tensors.items():
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
        data_parts.append(data)
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    path.write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + b"".join(data_parts))


def test_safetensors_shard_reads_tensor_offsets() -> None:
    with tempfile.TemporaryDirectory() as td:
        shard_path = Path(td) / "model-00001-of-00001.safetensors"
        tensor_bytes = struct.pack("<2f", 1.25, -2.5)
        write_safetensors(
            shard_path,
            {
                "language_model.model.layers.0.self_attn.q_proj.weight": ("F32", [2], tensor_bytes),
                "language_model.model.layers.0.mlp.down_proj.weight": ("F32", [1], struct.pack("<f", 3.0)),
            },
        )

        shard = SafetensorsShard(shard_path)
        ref = shard.tensor_ref("language_model.model.layers.0.self_attn.q_proj.weight")

        assert ref.dtype == "F32"
        assert ref.shape == (2,)
        assert ref.nbytes == 8
        assert shard.read_tensor_bytes(ref.name) == tensor_bytes


def test_gemma_weight_index_groups_layer_tensors() -> None:
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        shard_path = root / "model-00001-of-00001.safetensors"
        write_safetensors(
            shard_path,
            {
                "language_model.model.layers.0.self_attn.q_proj.weight": ("F32", [1], struct.pack("<f", 1.0)),
                "language_model.model.layers.0.mlp.down_proj.weight": ("F32", [1], struct.pack("<f", 2.0)),
                "language_model.model.layers.0.input_layernorm.weight": ("F32", [1], struct.pack("<f", 3.0)),
                "language_model.model.layers.1.router.weight": ("F32", [1], struct.pack("<f", 4.0)),
            },
        )
        index_path = root / "model.safetensors.index.json"
        index_path.write_text(
            json.dumps(
                {
                    "metadata": {"total_size": shard_path.stat().st_size},
                    "weight_map": {
                        name: shard_path.name
                        for name in SafetensorsShard(shard_path).header().keys()
                    },
                }
            ),
            encoding="utf-8",
        )

        weights = GemmaWeightIndex.from_model_dir(root)
        groups = weights.tensor_groups_for_layer(0)
        summary = weights.summary(require_shards=True)

        assert summary["tensor_count"] == 4
        assert len(groups["attention"]) == 1
        assert len(groups["mlp"]) == 1
        assert len(groups["norm"]) == 1
        assert weights.read_tensor_bytes("language_model.model.layers.1.router.weight") == struct.pack("<f", 4.0)


if __name__ == "__main__":
    test_safetensors_shard_reads_tensor_offsets()
    test_gemma_weight_index_groups_layer_tensors()
