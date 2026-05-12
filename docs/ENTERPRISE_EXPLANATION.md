# Enterprise Explanation

## What This Model Is

LLM_ccq4 is a packaged research LLM runtime built from three parts:

1. A Gemma-derived tensor set converted to `.ccq4`.
2. A Python orchestration layer that implements the token-forward loop.
3. A custom AMD OpenCL driver DLL that executes quantized matrix-vector operations and exposes additional enterprise/research kernels.

The goal is not to imitate the standard HuggingFace Transformers stack. The goal is to move the model into a custom substrate where weights can become persistent GPU-resident objects and where additional OpenCL kernels can later operate as inter-layer control, routing, noise, or simulation fields.

## Runtime Stack

```text
Prompt text
  -> Gemma tokenizer
  -> token ids
  -> embedding row lookup from CCQ4
  -> N transformer layers
     -> RMSNorm
     -> Q/K/V CCQ4 matvec
     -> q_norm/k_norm
     -> RoPE
     -> grouped attention over KV cache
     -> O projection CCQ4 matvec
     -> residual
     -> feed-forward RMSNorm
     -> gate/up/down CCQ4 matvec
     -> GELU and residual
  -> final RMSNorm
  -> tied embedding logits
  -> top-k/top-p/repetition-aware sampler
  -> next token
```

## Driver Role

`CC_OpenCl.dll` owns the OpenCL device context, kernel compilation, buffer allocation and execution. The important LLM-specific path is the persistent CCQ4 registry:

- `cc_register_persistent_ccq4_weight`
- `cc_execute_resident_ccq4_matvec`
- `cc_release_persistent_weight`
- `cc_release_all_persistent_weights`
- `cc_get_persistent_weight_count`

With `--preload-resident-layers`, the runtime registers all active layer matrices before token generation. For 10 layers this is 70 matrices:

```text
10 layers * (q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj)
```

This removes repeated weight upload from the generation loop. The remaining major cost is per-matvec host orchestration: input vector upload, output readback and individual kernel dispatch.

## What the CCQ4 Weights Contain

Each `.ccq4` file stores one tensor from the converted Gemma checkpoint. Examples:

- `model.language_model.embed_tokens.weight.ccq4`
- `model.language_model.layers.0.self_attn.q_proj.weight.ccq4`
- `model.language_model.layers.0.mlp.gate_proj.weight.ccq4`
- `model.language_model.norm.weight.ccq4`

The package includes 851 `.ccq4` tensor files and a manifest:

```text
model/ccq4/ccq4_full_language_manifest.json
```

The manifest records tensor names, shapes, dtypes and converted file references. It is the equivalent of an index file for the CCQ4 runtime.

## Why This Is Different From Normal Transformers

A normal inference stack treats the GPU backend as a generic matrix engine. This project exposes the driver and kernel layer as part of the model runtime. That enables:

- Explicit persistent weight residency.
- Driver-level noise and metrics feedback.
- Future fused OpenCL layer kernels.
- Future inter-layer substrate kernels such as Mycel routing, Ising relaxation, reservoir dynamics or attention-field control.

At the current stage, those substrate kernels are compiled and available in the driver, but the prompt-testable LLM path uses the transformer-compatible CCQ4 forward loop.

## Current Practical Baseline

Recommended prompt-test command:

```powershell
.\run_chat.ps1
```

Recommended engineering baseline:

```text
--gpu 1
--max-layers 10
--max-new-tokens 8
--preload-resident-layers
--vocab-limit 8192
--temperature 0.0
--no-repeat-window 12
```

This is intentionally conservative. It proves the package is complete, the driver loads, the weights become resident, and the forward loop can produce tokens.

## Next Engineering Targets

1. Persistent GPU input/output buffers for matvec calls.
2. Fused per-layer OpenCL forward kernel.
3. GPU-side logits over larger vocabulary chunks.
4. Slot-local driver shutdown for true simultaneous multi-GPU operation.
5. Full integration of substrate kernels as controllable inter-layer modifiers.

