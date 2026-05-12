$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

$env:PYTHONPATH = Join-Path $root "runtime"

python -m gemma_runtime.enterprise_model `
  --interactive `
  --quiet-driver `
  --driver-log logs\enterprise_driver.log `
  --preload-resident-layers `
  --system-prompt "Du bist das Sovereign-CC-Modell. Antworte knapp und technisch." `
  --tokenizer-json model\tokenizer\tokenizer.json `
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
