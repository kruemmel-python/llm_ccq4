# Model Card: LLM_ccq4 Sovereign-CC Gemma CCQ4

## Summary

LLM_ccq4 is a research packaging of a Gemma-derived language model substrate converted into the custom CCQ4 tensor format and executed through the CC OpenCL Enterprise driver. The package is designed for local AMD OpenCL experimentation on constrained hardware, especially 4 GB-class GPUs/APUs.

## Base Model

- Source family: Google Gemma 3n E4B local checkpoint.
- Converted artifact: CCQ4 tensor set under `model/ccq4`.
- Tokenizer: Gemma tokenizer files under `model/tokenizer`.
- Runtime architecture: Gemma-style transformer forward loop implemented in `runtime/gemma_runtime`.

## Intended Use

- Research into low-memory quantized inference.
- Driver-level GPU residency experiments.
- Hybrid model/runtime experiments combining language-model tensors with custom OpenCL kernels.
- Local prompt tests and profiling.

## Not Intended For

- Production user-facing assistant deployment.
- Safety-critical, legal, medical, or financial decision systems.
- Claims of full HuggingFace Transformers compatibility.
- Public redistribution without checking upstream Gemma terms.

## Architecture

The runtime uses:

- Tokenizer and embeddings from the Gemma checkpoint.
- Per-token autoregressive forward loop.
- RMSNorm, RoPE, grouped attention and KV cache.
- MLP gate/up/down projection with GELU activation.
- Final norm and embedding-tied logits.
- CCQ4 quantized weights loaded by a custom Python reader.
- `CC_OpenCl.dll` for OpenCL GPU matvec execution.

The current packaged runtime can execute a configurable prefix of the full layer stack using `--max-layers`. The 10-layer configuration is the practical prompt-test baseline for this package.

## Quantization

Weights are stored as `.ccq4` files. CCQ4 is a blockwise 4-bit signed quantization container with per-block scale metadata. It is optimized for simple streaming decode and custom OpenCL matvec kernels. See [docs/CCQ4_FORMAT.md](docs/CCQ4_FORMAT.md).

## Hardware Notes

The package is tuned around AMD OpenCL:

- `--gpu 0` usually selects the integrated `gfx90c` device.
- `--gpu 1` usually selects the faster `gfx1034` device when available.
- 10 layers preload 70 resident matrices: 4 attention projections and 3 MLP projections per layer.

## Known Limitations

- The current implementation is slow for long outputs because orchestration remains Python/host driven.
- Full-model 35-layer inference requires substantial time on the tested hardware.
- The quality of generated text is not yet equivalent to a production Gemma runtime.
- Sampling limits such as `--no-repeat-window` are used to reduce short-loop degeneration.

## Ethical and Legal Notes

This package contains or references converted model weights derived from Gemma. Upstream model terms apply. Review the Gemma license and acceptable use policy before distribution or deployment.

