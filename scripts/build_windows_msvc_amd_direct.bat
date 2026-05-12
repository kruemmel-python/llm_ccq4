@echo off
setlocal enabledelayedexpansion

rem Direct MSVC/AMD OpenCL build for CC_OpenCl.dll inside the LLM_ccq4 package.
rem Run this from "x64 Native Tools Command Prompt for VS 2022".
rem No MinGW, no Ninja, no CMake required.

cd /d "%~dp0\.."

if not exist driver\build mkdir driver\build

where cl >nul 2>nul
if errorlevel 1 (
    echo Fehler: cl.exe nicht gefunden.
    echo Starte "x64 Native Tools Command Prompt for VS 2022" und fuehre dieses Skript dort aus.
    exit /b 1
)

where lib >nul 2>nul
if errorlevel 1 (
    echo Fehler: lib.exe nicht gefunden.
    echo Starte "x64 Native Tools Command Prompt for VS 2022" und fuehre dieses Skript dort aus.
    exit /b 1
)

rem Install/copy official Khronos OpenCL headers into .\driver\CL when missing.
if not exist driver\CL\cl.h (
    echo OpenCL-Headers fehlen. Versuche KhronosGroup/OpenCL-Headers per git zu holen...
    where git >nul 2>nul
    if errorlevel 1 (
        echo Fehler: git nicht gefunden und driver\CL\cl.h fehlt.
        echo Lade OpenCL-Headers als ZIP von KhronosGroup/OpenCL-Headers und kopiere den Ordner CL nach:
        echo %CD%\driver\CL
        exit /b 1
    )
    if not exist _OpenCL-Headers (
        git clone https://github.com/KhronosGroup/OpenCL-Headers.git _OpenCL-Headers
        if errorlevel 1 exit /b 1
    )
    xcopy /E /I /Y _OpenCL-Headers\CL driver\CL
    if errorlevel 1 exit /b 1
)

if not exist driver\CL\cl_platform.h (
    echo Fehler: driver\CL\cl_platform.h fehlt. Der CL-Headerordner ist unvollstaendig.
    exit /b 1
)

rem Create/import OpenCL.lib from AMD/SDK OpenCL.def if no import library exists yet.
if not exist driver\build\OpenCL.lib (
    if exist driver\CL\OpenCL.def (
        lib /def:driver\CL\OpenCL.def /machine:x64 /out:driver\build\OpenCL.lib
        if errorlevel 1 exit /b 1
    ) else (
        echo Fehler: driver\build\OpenCL.lib fehlt und driver\CL\OpenCL.def wurde nicht gefunden.
        echo Kopiere deine AMD/OpenCL-Dateien nach .\driver\CL, sodass z.B. existiert:
        echo   driver\CL\OpenCL.def
        echo Alternativ lege eine passende x64 OpenCL.lib unter driver\build\OpenCL.lib ab.
        exit /b 1
    )
)

echo.
echo === Build CC_OpenCl.dll mit MSVC/C++17 und OpenCL 3.0 Headers ===
cl /LD /O2 /TP /std:c++17 ^
 /DCL_TARGET_OPENCL_VERSION=300 ^
 /DCL_USE_DEPRECATED_OPENCL_1_2_APIS ^
 /D_CRT_SECURE_NO_WARNINGS ^
 /Idriver\include ^
 /Idriver ^
 /Idriver\CL ^
 driver\src\CC_OpenCL.c driver\src\CipherCore_NoiseCtrl.c ^
 /Fe:driver\build\CC_OpenCl.dll ^
 /link driver\build\OpenCL.lib

if errorlevel 1 (
    echo.
    echo Build fehlgeschlagen.
    exit /b 1
)

echo.
echo Build erfolgreich:
dir driver\build\CC_OpenCl.*
