# Build and Package Guide

## Build Environment

Recommended:

- Windows 10/11 x64.
- Visual Studio 2022 Build Tools.
- AMD OpenCL runtime.
- PowerShell.
- Python 3.10+.

No CMake, Ninja, MinGW or Rust toolchain is required for the inference path.

## Build `CC_OpenCl.dll`

Open "x64 Native Tools Command Prompt for VS 2022" or otherwise ensure `cl.exe` and `lib.exe` are available.

```powershell
cd D:\CC_OpenCl_Driver_Enterprise\LLM_ccq4
.\build_driver.ps1
```

Output:

```text
driver/build/CC_OpenCl.dll
driver/build/CC_OpenCl.lib
driver/build/CC_OpenCl.exp
```

## Run a Smoke Test

```powershell
.\run_smoke.ps1
```

Expected result includes:

```json
"preload": {
  "enabled": true,
  "matrix_count": 7,
  "resident_matrix_count": 7
}
```

For a 10-layer interactive run, expect:

```json
"resident_matrix_count": 70
```

## Rebuild CCQ4 Weights

If you have the source Gemma checkpoint locally:

```powershell
$env:PYTHONPATH="runtime"
python -m gemma_runtime.build_full_model `
  --model-dir D:\Models\gemma-3n-E4B `
  --output-dir model\ccq4
```

This recreates `.ccq4` files from local safetensors. It is not required for normal use of this package because the converted files are already included.

## Package Contents Checklist

Before publishing or moving the folder, verify:

```powershell
Get-ChildItem model\ccq4 -Filter *.ccq4 | Measure-Object
Test-Path driver\build\CC_OpenCl.dll
Test-Path model\tokenizer\tokenizer.json
Test-Path runtime\gemma_runtime\enterprise_model.py
```

