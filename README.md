# LLM_ccq4

LLM_ccq4 is a local, Windows/AMD OpenCL distribution of the Sovereign-CC Gemma CCQ4 runtime. It packages:

- CCQ4 quantized Gemma-derived tensor weights.
- A dependency-light Python inference runtime.
- The compiled `CC_OpenCl.dll`.
- The complete C/OpenCL driver sources and MSVC build script.
- Tokenizer/config files needed for local prompt tests.
- Enterprise documentation explaining the model, the driver, and the `.ccq4` format.

This is an experimental research runtime, not a drop-in HuggingFace Transformers model. It executes a partial Gemma-style autoregressive forward loop through the custom CC OpenCL driver with persistent CCQ4 weight residency.

## Quick Start

From PowerShell:

```powershell
cd D:\CC_OpenCl_Driver_Enterprise\LLM_ccq4
.\run_smoke.ps1
```

Interactive prompt test:

```powershell
.\run_chat.ps1
```

Manual command:

```powershell
$env:PYTHONPATH="runtime"
python -m gemma_runtime.enterprise_model `
  --interactive `
  --quiet-driver `
  --driver-log logs\enterprise_driver.log `
  --preload-resident-layers `
  --ccq4-dir model\ccq4 `
  --dll driver\build\CC_OpenCl.dll `
  --gpu 1 `
  --max-layers 10 `
  --max-new-tokens 8 `
  --emotion-mode precise `
  --temperature 0.0 `
  --top-k 16 `
  --vocab-limit 8192 `
  --no-repeat-ngram-size 2 `
  --no-repeat-window 12 `
  --repetition-penalty 1.25
```

## Build the Driver

Use a Visual Studio 2022 x64 Native Tools shell, or run from a shell where `cl.exe` and `lib.exe` are on `PATH`:

```powershell
.\build_driver.ps1
```

The output DLL is written to:

```text
driver\build\CC_OpenCl.dll
```

## Directory Layout

```text
LLM_ccq4/
  model/
    ccq4/                 Quantized CCQ4 weights and manifest
    tokenizer/            Gemma tokenizer/config files
  runtime/gemma_runtime/  Python runtime, quantizer, tokenizer, forward loop
  driver/
    src/                  C/OpenCL driver sources
    include/              Driver API headers
    CL/                   OpenCL headers/import helpers
    build/                Prebuilt CC_OpenCl.dll and import libs
  scripts/                Build and model utility scripts
  tests/                  Focused runtime tests
  docs/                   Enterprise documentation
```

## Current Runtime Status

Known working:

- CCQ4 tensor loading.
- Persistent CCQ4 weight registration.
- GPU resident matvec via `CC_OpenCl.dll`.
- Gemma-style token forward loop: embedding lookup, RMSNorm, RoPE, grouped attention with KV cache, MLP/GELU, residuals, final norm, logits.
- Interactive session with system prompt and repetition controls.

Current limitations:

- The runtime is still host-orchestrated. Each token performs many individual matvec calls.
- Full 35-layer inference is expensive on 4 GB-class hardware.
- `--gpu 1` selects the faster discrete AMD device when present. `--also-gpu` is only a probe path in the current driver lifecycle.
- Text quality is research-stage and depends strongly on layer count, vocabulary limit, and sampling controls.

## License and Model Terms

The CC driver/runtime source and the converted weights have separate provenance. The `.ccq4` weights are derived from Google Gemma model tensors and must be used only under the applicable upstream Gemma terms and any access terms that applied to the source model. Do not redistribute this package publicly unless those terms permit your intended distribution.

