$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

cmd.exe /c "scripts\build_windows_msvc_amd_direct.bat"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}
