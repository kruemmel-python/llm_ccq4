# Pipeline Recipes

Version: Enterprise Agent-Ready  
Stand: 2026-05-06

Dieses Dokument enthält sichere, wiederverwendbare Ablaufmuster für Hostprogramme und KI-Agenten. Jede Pipeline enthält Zweck, Buffer, Sequenz, VRAM-Hinweise und typische Fehler.

## Rezept 0: Gemma CCQ4 Persistent Substrate

### Zweck
Starte das prompt-testbare Gemma-CC-Modell mit residenten CCQ4-Gewichten. `--preload-resident-layers` registriert alle aktiven Layer-Matrizen vor dem ersten Token; bei 10 Layern mit Attention + MLP sind das 70 residente Matrizen.

### Sequenz

```powershell
$env:PYTHONPATH="subqg_driver_tests"
python -m gemma_runtime.enterprise_model `
  --interactive `
  --quiet-driver `
  --driver-log build\gemma_full_ccq4\enterprise_driver.log `
  --preload-resident-layers `
  --system-prompt "Du bist das Sovereign-CC-Modell. Antworte knapp und technisch." `
  --ccq4-dir build\gemma_full_ccq4 `
  --dll build\CC_OpenCl.dll `
  --gpu 1 `
  --max-layers 10 `
  --max-new-tokens 16 `
  --emotion-mode precise `
  --top-k 64 `
  --vocab-limit 8192 `
  --no-repeat-ngram-size 2 `
  --no-repeat-window 8
```

### GPU-Regel
`--gpu 1` nutzt die schnelle `gfx1034` als Hauptinferenzgerät. `--also-gpu 1` ist nur eine Co-GPU-Residency-Probe und beendet die Probe wieder, weil der aktuelle Treiber-Shutdown noch globale OpenCL-Ressourcen freigibt.

### Diagnose
Die Konsolenzusammenfassung enthält `preload.matrix_count` und `preload.resident_matrix_count`. Für 10 Layer muss `resident_matrix_count` nach dem Preload 70 erreichen, bevor die eigentliche Token-Schleife beginnt.

### Sampling-Regel
`--no-repeat-ngram-size 2` blockiert wiederholte Bigramme. `--no-repeat-window 8` blockiert zusätzlich die letzten acht Token vollständig und verhindert kurze Degenerationsschleifen wie `token A, token B, token A`.

## Rezept 1: Resonance + Energy Scheduler

### Zweck
Erzeuge ein gekoppeltes Aktivitätsfeld und extrahiere daraus aktive Agenten/Zellen.

### Buffer

| Buffer | Typ | Größe | Richtung |
|---|---|---:|---|
| state | float | N | in/out |
| velocity | float | N | in/out |
| drive | float | N | in |
| energy | float | N | out/in für Scheduler |
| neighbors | int | N*K | in |
| weights | float | N*K | in |
| active_flags | uchar | N | out |
| active_indices | int | N | out |
| active_count | uint | 1 | in/out, vor Aufruf 0 |

### Sequenz

```pseudo
initialize_gpu(gpu)

allocate state, velocity, drive, energy, neighbors, weights
allocate active_flags, active_indices, active_count

write initial state, velocity, drive, neighbors, weights

loop:
    execute_resonant_field_step_gpu(...)
    write active_count = 0
    execute_energy_gated_scheduler_gpu(...)
    read active_count

    if active_count == 0:
        pause or increase drive
    elif active_count == N:
        increase threshold or skip scheduler
    else:
        process active_indices
```

### Stabilitätsregeln

```text
dt <= 0.05 für neue Systeme
coupling <= 0.25 für Standardläufe
clamp_abs > 0 für Agentenbetrieb
```

## Rezept 2: Mycelium + Morphogenesis

### Zweck
Nährstoff-/Pheromondynamik treibt Zelltypänderungen.

### Sequenz

```pseudo
initialize_gpu
subqg_init_mycel(gpu, T_cap=N, C=4, K=8)
set_neighbors_sparse
set_nutrient_state
set_mood_state

loop:
    step_pheromone_reinforce
    step_pheromone_diffuse_decay
    step_mycel_update
    read or map nutrient/energy fields if needed
    execute_morphogenetic_rule_step_gpu
    step_colony_update
```

### Agentenhinweis
Morphogenese-Regeln sind deterministisch in Tabellenreihenfolge. Ein Agent muss Regeländerungen versionieren, damit Simulationsergebnisse reproduzierbar bleiben.

## Rezept 3: SubQG Multifield Checkpoint

### Zweck
Lade oder sichere acht Feldkanäle: energy, pressure, gravity, magnetism, temperature, potential, drift_x, drift_y.

### Sequenz

```pseudo
subqg_set_multifield_state(gpu, N, energy, pressure, gravity, magnetism,
                           temperature, potential, drift_x, drift_y)

loop:
    subqg_simulation_step_batched(...)
    if checkpoint_due:
        subqg_get_multifield_state(...)
```

### VRAM
`8 * N * sizeof(float)` plus Runtime-Overhead. Bei N=1.048.576 sind das 32 MiB.

## Rezept 4: Quantum-Isolated Run

### Zweck
Führe quantenähnliche Algorithmen aus, ohne Mycelium/Bio-Brain VRAM zu blockieren.

### Sequenz

```pseudo
initialize_gpu
disable or release large field domains
execute_grover_gpu / execute_vqe_gpu / execute_qaoa_gpu
read output distribution
shutdown_gpu or release quantum state
```

### Agentenregel
Quantum Statevector ist exponentiell. Ein Agent darf q nur erhöhen, wenn das VRAM-Budget neu berechnet wurde.

## Rezept 5: Bio-Brain + Resonance Attention

### Zweck
Nutze Resonanzenergie als Aufmerksamkeitssignal für neuronale oder agentische Updates.

### Sequenz

```pseudo
initialize_gpu
allocate resonance buffers
allocate bio-brain neuron/synapse buffers
execute_resonant_field_step_gpu
execute_energy_gated_scheduler_gpu
if active_count / N < 0.2:
    process active subset
else:
    process full brain step
```

### Rückfall
Wenn aktive Rate > 80 %, Scheduler deaktivieren. Der Scan plus indirekte Zugriffe sind dann meist teurer als ein Vollupdate.

## Einheitliche Fehlerbehandlung

```pseudo
status = call_driver_function(...)
if status != 1:
    err = cc_get_last_error()
    if contains(err, "CL_OUT_OF_RESOURCES"):
        reduce_grid_or_disable_domain()
    elif contains(err, "CL_INVALID_MEM_OBJECT"):
        recreate_buffers()
    elif contains(err, "CL_INVALID_KERNEL_ARGS"):
        verify_contract()
    else:
        abort_with_diagnostics(err)
```

## Rezept 4: Thermodynamic Langevin Relaxation

### Zweck
Führe probabilistische Relaxation eines Feldes aus und nutze `free_energy` als Diagnose- oder Scheduler-Signal.

### Buffer

| Buffer | Typ | Größe | Richtung |
|---|---|---:|---|
| state | float | N | in/out |
| momentum | float | N | in/out |
| bias | float | N | in |
| free_energy | float | N | out |
| neighbors | int | N*K | in |
| weights | float | N*K | in |

### Sequenz

```pseudo
write state, momentum, bias, neighbors, weights
execute_thermodynamic_langevin_step_gpu(...)
finish_gpu
read free_energy only if host diagnostics are required
```

### Stabilitätsregel

```pseudo
if unstable:
    dt *= 0.5
    temperature *= 0.5
    coupling *= 0.5
    clamp_abs = max(clamp_abs, 4.0)
```

## Rezept 5: E-I Plastic Reservoir Double Buffer

### Zweck
Erzeuge eine adaptive, nichtlineare Agenten-/Neuronensignaldynamik.

### Sequenz

```pseudo
signal = buffer_A
next_signal = buffer_B

loop:
    execute_ei_plastic_reservoir_step_gpu(signal, next_signal, ...)
    swap(signal, next_signal)
```

`gain_e`, `gain_i` und `activity_ema` bleiben persistent und werden nicht pro Tick neu initialisiert.

## Rezept 6: Tensor Bond Entropy Gate als GPU-Arbeitsliste

### Zweck
Verarbeite nur Bonds mit hoher Unsicherheit oder starker Änderung.

### Sequenz

```pseudo
active_count[0] = 0
write active_count
execute_tensor_bond_entropy_gate_gpu(...)
read active_count

if active_count > 0:
    process active_bonds[0:active_count]
```

### Agentenregel

```pseudo
if active_count / B > 0.8:
    Gate ist zu offen
if active_count / B < 0.001:
    Gate ist zu streng
```
