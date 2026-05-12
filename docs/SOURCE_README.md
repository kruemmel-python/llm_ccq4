# CC_OpenCl Driver

Standalone-Projekt fuer den kompilierbaren `CC_OpenCl` OpenCL-Treiber.

Dieses Paket ist fuer zukuenftige Projekte als eigenstaendiger DLL-Build vorbereitet.

## Wichtige Aenderung in dieser Version

`src/CC_OpenCL.c` enthaelt eingebettete OpenCL-Kernel als C++ Raw-String-Literale:

```cpp
R"CLC(
...
)CLC"
```

Darum muss die Datei unter MSVC als C++ kompiliert werden, obwohl sie die Endung `.c` hat. Der direkte Windows-Build nutzt deshalb:

```cmd
/TP /std:c++17
```

Zusätzlich wurden die vier SymBio-API-Prototypen in `include/SymBio_Interface.h` mit `__declspec(dllexport)` kompatibel zur Implementierung gemacht.

## Empfohlen: Windows/MSVC/AMD ohne MinGW und ohne CMake

Voraussetzungen:

- Visual Studio Build Tools 2022
- Komponente: `Desktop development with C++`
- AMD-Grafiktreiber mit OpenCL-Runtime
- `CL\OpenCL.def` aus deinem AMD/OpenCL-SDK oder eine passende `build\OpenCL.lib`
- optional `git`, damit das Skript die Khronos OpenCL-Headers automatisch holen kann

Starte:

```text
x64 Native Tools Command Prompt for VS 2022
```

Dann:

```cmd
cd /d D:\CC_OpenCl_Driver
scripts\build_windows_msvc_amd_direct.bat
```

Das Skript:

1. prueft `cl.exe` und `lib.exe`,
2. holt bei Bedarf die offiziellen Khronos OpenCL-Headers nach `.\CL`,
3. erzeugt `build\OpenCL.lib` aus `CL\OpenCL.def`,
4. baut `build\CC_OpenCl.dll`.

Erwartetes Ergebnis:

```text
build\CC_OpenCl.dll
build\CC_OpenCl.lib
build\CC_OpenCl.exp
```

## Manuelle Befehle fuer AMD/MSVC

Falls du alles manuell ausfuehren willst:

```cmd
cd /d D:\CC_OpenCl_Driver

git clone https://github.com/KhronosGroup/OpenCL-Headers.git _OpenCL-Headers
xcopy /E /I /Y _OpenCL-Headers\CL CL

mkdir build
lib /def:CL\OpenCL.def /machine:x64 /out:build\OpenCL.lib

cl /LD /O2 /TP /std:c++17 ^
 /DCL_TARGET_OPENCL_VERSION=300 ^
 /DCL_USE_DEPRECATED_OPENCL_1_2_APIS ^
 /D_CRT_SECURE_NO_WARNINGS ^
 /Iinclude ^
 /I. ^
 src\CC_OpenCL.c src\CipherCore_NoiseCtrl.c ^
 /Fe:build\CC_OpenCl.dll ^
 /link build\OpenCL.lib
```

Wichtig: Nicht `CL\opencl.h` nach `CL\cl.h` kopieren. `opencl.h` ist ein Umbrella-Header; `cl.h` muss der echte Khronos-Header sein.

## CMake

`CMakeLists.txt` ist ebenfalls angepasst: Das Projekt aktiviert C++17 und kompiliert die `.c`-Dateien als C++.

Ein MSVC-CMake-Build funktioniert nur mit einer Windows-CMake-Installation, die Generatoren wie `NMake Makefiles` oder `Visual Studio 17 2022` kennt. In deiner Umgebung war `cmake.exe` eine Unix/MSYS-Variante; deshalb ist der direkte MSVC-Build oben robuster.

## Laufzeit

Zur Laufzeit muss eine OpenCL-Runtime vorhanden sein. Bei AMD kommt diese normalerweise mit dem AMD-Grafiktreiber. Prüfen:

```cmd
dir C:\Windows\System32\OpenCL.dll
```

## Dateien

```text
src/CC_OpenCL.c
src/CipherCore_NoiseCtrl.c
include/CipherCore_NoiseCtrl.h
include/SymBio_Interface.h
scripts/build_windows_msvc_amd_direct.bat
CMakeLists.txt
README.md
```

## Fehlerbilder

### `"CL/cl.h": No such file or directory`

OpenCL-Headers fehlen. Das Skript kann sie per `git` holen, oder du kopierst den Ordner `CL` aus `KhronosGroup/OpenCL-Headers` manuell in das Projekt.

### `OpenCL.lib kann nicht geöffnet werden`

`build\OpenCL.lib` fehlt. Erzeuge sie aus deiner `CL\OpenCL.def`:

```cmd
lib /def:CL\OpenCL.def /machine:x64 /out:build\OpenCL.lib
```

### `Zeilenvorschub in Konstante` oder Fehler bei `__kernel`

Dann wurde `CC_OpenCL.c` als C statt C++ kompiliert. Nutze `/TP /std:c++17`.

### `Neudefinition; unterschiedliche Bindung`

Dann ist eine alte `include\SymBio_Interface.h` im Einsatz. Diese Version enthält `SYMBIO_API`/`__declspec(dllexport)` an den relevanten Prototypen.


## Enterprise Algorithm Pack

Diese Version enthält eine zusätzliche, modular kompilierte Algorithmus-Schicht in `src/CC_OpenCL.c`. Sie ist absichtlich vom historischen Monolith-Kernel getrennt und wird lazy beim ersten Aufruf kompiliert.

### Neue Exportfunktionen

#### `execute_resonant_field_step_gpu`

Gekoppeltes Oszillator-/Resonanzfeld über ein fixes Nachbarschaftslayout.

```c
int execute_resonant_field_step_gpu(
    int gpu_index,
    void* state_buf,
    void* velocity_buf,
    void* drive_buf,
    void* energy_buf,
    void* neighbors_buf,
    void* weights_buf,
    int N,
    int K,
    float dt,
    float damping,
    float coupling,
    float inertia,
    float clamp_abs);
```

Buffer-Layout:

- `state_buf`: `float[N]`
- `velocity_buf`: `float[N]`
- `drive_buf`: `float[N]`
- `energy_buf`: `float[N]`
- `neighbors_buf`: `int[N * K]`
- `weights_buf`: `float[N * K]`

#### `execute_energy_gated_scheduler_gpu`

GPU-seitiges Aktivitäts-Gating. Der Treiber erzeugt `active_flags[N]`, kompaktiert aktive Indizes in `active_indices[N]` und setzt `active_count[0]`.

```c
int execute_energy_gated_scheduler_gpu(
    int gpu_index,
    void* energy_buf,
    void* nutrient_buf,
    void* active_flags_buf,
    void* active_indices_buf,
    void* active_count_buf,
    int N,
    float threshold,
    float sleep_decay,
    float nutrient_recovery);
```

Buffer-Layout:

- `energy_buf`: `float[N]`
- `nutrient_buf`: `float[N]`
- `active_flags_buf`: `unsigned char[N]`
- `active_indices_buf`: `int[N]`
- `active_count_buf`: `uint[1]`

`active_count_buf` wird vor jeder Markierung vom Treiber auf 0 gesetzt.

#### `execute_morphogenetic_rule_step_gpu`

Tabellarische Morphogenese-Regeln auf GPU-Feldern.

```c
int execute_morphogenetic_rule_step_gpu(
    int gpu_index,
    void* cell_type_buf,
    void* nutrient_buf,
    void* energy_buf,
    void* potential_buf,
    void* rule_in_type_buf,
    void* rule_min_nutrient_buf,
    void* rule_min_energy_buf,
    void* rule_out_type_buf,
    void* rule_delta_potential_buf,
    int N,
    int R,
    float nutrient_cost);
```

Buffer-Layout:

- `cell_type_buf`: `unsigned char[N]`
- `nutrient_buf`: `float[N]`
- `energy_buf`: `float[N]`
- `potential_buf`: `float[N]`
- `rule_in_type_buf`: `int[R]`
- `rule_min_nutrient_buf`: `float[R]`
- `rule_min_energy_buf`: `float[R]`
- `rule_out_type_buf`: `int[R]`
- `rule_delta_potential_buf`: `float[R]`

### Architekturhinweis

Der neue Pack kompiliert intern ein eigenes `cl_program` und eigene `cl_kernel`-Handles:

- `cc_resonant_field_step`
- `cc_mark_active_agents`
- `cc_gated_agent_decay`
- `cc_morphogenetic_rule_step`

Dadurch werden neue Forschungsalgorithmen nicht mehr direkt in den bestehenden großen Kernel-String eingebettet. Build-Fehler, Kernel-Handles und Cleanup sind klar isoliert.

