$ErrorActionPreference = "Stop"

cmake -S . -B build-mingw -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build-mingw

Write-Host "Built: build-mingw/bin/CC_OpenCl.dll"
