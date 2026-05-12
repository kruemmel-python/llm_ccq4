$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$env:PYTHONPATH = Join-Path $root "runtime"

python -m gemma_runtime.enterprise_model `
  --quiet-driver `
  --driver-log logs\enterprise_smoke_driver.log `
  --preload-resident-layers `
  --prompt "[2]" `
  --tokenizer-json model\tokenizer\tokenizer.json `
  --ccq4-dir model\ccq4 `
  --dll driver\build\CC_OpenCl.dll `
  --gpu 1 `
  --max-layers 1 `
  --max-new-tokens 0 `
  --top-k 4 `
  --vocab-limit 128
