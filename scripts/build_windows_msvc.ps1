$ErrorActionPreference = "Stop"

# MSVC/AMD direct build. This does not require MinGW, Ninja or CMake.
# Run from "Developer PowerShell for VS 2022" or "x64 Native Tools Command Prompt for VS 2022".
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    cmd /c scripts\build_windows_msvc_amd_direct.bat
    if ($LASTEXITCODE -ne 0) { throw "Build failed with exit code $LASTEXITCODE" }
} finally {
    Pop-Location
}
