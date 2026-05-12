# CCQ4 Format

## Purpose

CCQ4 is the custom weight container used by LLM_ccq4. It stores tensors in blockwise 4-bit quantized form so that model weights can be kept small enough for 4 GB-class AMD OpenCL devices and loaded into the custom driver as persistent GPU-resident matrices.

## High-Level Structure

Each `.ccq4` file represents one tensor. The runtime opens these files directly; there is no PyTorch or safetensors dependency during inference.

Conceptually each file contains:

- Tensor metadata: shape, rows, columns and block layout.
- Quantized payload: packed 4-bit signed values.
- Per-block scale values used to reconstruct approximate float values.

The Python implementation is in:

```text
runtime/gemma_runtime/quantizer.py
```

The GPU matvec path is in:

```text
runtime/gemma_runtime/matvec.py
driver/src/CC_OpenCL.c
```

## Quantization Method

The converter reads tensors from the source safetensors checkpoint, partitions values into fixed-size blocks, computes a scale per block, then stores signed 4-bit integer values. During matvec, the values are dequantized logically as:

```text
float_value ~= int4_value * block_scale
```

This is lossy quantization. It trades exact reconstruction for reduced disk size, reduced transfer size and practical residency on constrained GPU memory.

## Why Blockwise Scaling

One scale for an entire tensor is usually too coarse. Per-value scaling is too expensive. Blockwise scaling gives a pragmatic middle point:

- Local dynamic range is preserved better than global scaling.
- Metadata overhead is small.
- GPU kernels can stream blocks efficiently.
- The format stays simple enough to debug.

## What Is Included In This Package

The packaged CCQ4 set contains:

- 851 `.ccq4` tensor files.
- `ccq4_full_language_manifest.json`.
- Embedding tensors.
- All language model layer attention projections.
- All language model MLP projections.
- Norm vectors and additional Gemma-3n structural tensors.

The current prompt-test runtime actively uses:

- `embed_tokens.weight`
- layer `self_attn.q_proj/k_proj/v_proj/o_proj`
- layer `self_attn.q_norm/k_norm`
- layer input and feed-forward RMSNorm weights
- layer `mlp.gate_proj/up_proj/down_proj`
- final `norm.weight`

Additional converted tensors are retained so the package remains complete for future runtime expansion.

## Creation Pipeline

The full model was created from local Gemma safetensors with:

```powershell
$env:PYTHONPATH="subqg_driver_tests"
python -m gemma_runtime.build_full_model `
  --model-dir D:\Models\gemma-3n-E4B `
  --output-dir build\gemma_full_ccq4
```

The packaging step copies the generated `.ccq4` files into:

```text
model/ccq4/
```

## Manifest

`model/ccq4/ccq4_full_language_manifest.json` is the package index. It allows tooling to inspect the converted tensor set without scanning every file deeply.

