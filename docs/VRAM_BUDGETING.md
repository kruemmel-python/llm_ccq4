# VRAM Budgeting und CL_OUT_OF_RESOURCES-Prävention

Version: Enterprise Agent-Ready  
Stand: 2026-05-06

Dieses Dokument definiert ein konservatives, agentenfähiges Speicherbudget für `CC_OpenCl.dll`. Ziel ist, `CL_OUT_OF_RESOURCES`, `CL_MEM_OBJECT_ALLOCATION_FAILURE`, instabile Treiberzustände und nicht reproduzierbare Allokationsfehler vor dem Kernelstart zu vermeiden.

## Standardparameter

| Parameter | Wert | Bedeutung |
|---|---:|---|
| N | 1,048,576 | Standard-Grid 1024 x 1024 |
| K | 8 | Nachbarn pro Zelle/Knoten |
| C | 4 | Pheromon-/Mood-Kanäle |
| R | 16 | Morphogenese-Regeln |
| Quantum q | 24 | Statevector-Größe 2^24 |
| Float | 4 Bytes | OpenCL `float` |
| Int | 4 Bytes | OpenCL `int` |
| UChar | 1 Byte | OpenCL `uchar` |

## Sicherheitsregel

Ein Agent oder Hostprogramm darf nicht den gesamten gemeldeten VRAM verplanen.

```text
safe_vram_bytes = CL_DEVICE_GLOBAL_MEM_SIZE * 0.65
aggressive_vram_bytes = CL_DEVICE_GLOBAL_MEM_SIZE * 0.80
required_bytes <= safe_vram_bytes
```

`0.65` ist der Enterprise-Default, weil OpenCL-Runtime, Treiber, Kernelprogramme, Command Queues, temporäre Buffer, Windows-GPU-Speicherverwaltung und andere Anwendungen zusätzlichen Speicher beanspruchen.

## Domänenübersicht bei Standard-Grid

| Domäne | Formel | Standardverbrauch |
|---|---|---|
| Mycelium Core | pheromone[N*C*K]*float + mood[N*C]*float + nutrient[N]*float + potential[N]*float + colonies[N]*uint8 + neighbors[N*K]*int | 185.00 MiB |
| SubQG Multifield | 8 scalar channels[N]*float | 32.00 MiB |
| Resonance Field | state/velocity/drive/energy[N]*float + neighbors[N*K]*int + weights[N*K]*float | 80.00 MiB |
| Energy Scheduler | active_flags[N]*uchar + active_indices[N]*int + active_count[1]*uint | 5.00 MiB |
| Morphogenesis | cell_type[N]*uchar + nutrient/energy/potential[N]*float + rule tables[R] | 13.00 MiB |
| Bio-Brain Standard | neurons=N, fanout=K, synapses=N*K: neuron scalar buffers + sparse synapse tables | 149.00 MiB |
| Quantum Statevector | complex amplitudes[2^q]*float2 + probability/phase scratch | 256.00 MiB |

## Skalierungstabelle

Alle Werte sind MiB. Annahmen: K=8, C=4, R=16. Quantum-Statevector ist separat zu planen, weil er nicht vom 2D-Grid abhängt.

| Grid | N | Mycel MiB | SubQG MiB | Resonance MiB | Scheduler MiB | Morph MiB | Bio-Brain MiB |
|---|---|---|---|---|---|---|---|
| 256x256 | 65,536 | 11.6 | 2.0 | 5.0 | 0.3 | 0.8 | 9.3 |
| 512x512 | 262,144 | 46.2 | 8.0 | 20.0 | 1.3 | 3.3 | 37.2 |
| 1024x1024 | 1,048,576 | 185.0 | 32.0 | 80.0 | 5.0 | 13.0 | 149.0 |
| 2048x2048 | 4,194,304 | 740.0 | 128.0 | 320.0 | 20.0 | 52.0 | 596.0 |
| 4096x4096 | 16,777,216 | 2960.0 | 512.0 | 1280.0 | 80.0 | 208.0 | 2384.0 |

## Kombinierte Beispielbudgets

| Pipeline | Enthaltene Domänen | Standard 1024x1024 | Empfehlung |
|---|---|---:|---|
| Mycel Solo | Mycelium Core | 185.0 MiB | Auch auf 2 GB GPUs möglich |
| Mycel + SubQG | Mycelium + SubQG | 217.0 MiB | 4 GB+ empfohlen |
| Resonance + Scheduler | Resonance + Scheduler | 85.0 MiB | Sehr gut für Agentensteuerung |
| Full Bio Field | Mycel + SubQG + Resonance + Scheduler + Morphogenesis | 315.0 MiB | 6 GB+ empfohlen |
| Bio-Brain + Resonance | Bio-Brain + Resonance + Scheduler | 234.0 MiB | 8 GB+ empfohlen |
| Quantum q=24 | Statevector + Scratch | 256.0 MiB | q erhöhen nur mit VRAM-Prüfung |

## Detailtabellen

### Mycelium Core

Standardannahme: N=1,048,576, K=8, C=4; Quantum Statevector nutzt q=24.

| Buffer | Typ | Elemente | Bytes/Element | Bytes | MiB |
|---|---|---|---|---|---|
| pheromone | float | N*C*K | 4 | 134,217,728 | 128.00 |
| mood | float | N*C | 4 | 16,777,216 | 16.00 |
| nutrient | float | N | 4 | 4,194,304 | 4.00 |
| potential | float | N | 4 | 4,194,304 | 4.00 |
| colonies | uint8 | N | 1 | 1,048,576 | 1.00 |
| neighbors_sparse | int | N*K | 4 | 33,554,432 | 32.00 |

**Gesamt:** 193,986,560 Bytes = **185.00 MiB**.

### SubQG Multifield

Standardannahme: N=1,048,576, K=8, C=4; Quantum Statevector nutzt q=24.

| Buffer | Typ | Elemente | Bytes/Element | Bytes | MiB |
|---|---|---|---|---|---|
| energy | float | N | 4 | 4,194,304 | 4.00 |
| pressure | float | N | 4 | 4,194,304 | 4.00 |
| gravity | float | N | 4 | 4,194,304 | 4.00 |
| magnetism | float | N | 4 | 4,194,304 | 4.00 |
| temperature | float | N | 4 | 4,194,304 | 4.00 |
| potential | float | N | 4 | 4,194,304 | 4.00 |
| drift_x | float | N | 4 | 4,194,304 | 4.00 |
| drift_y | float | N | 4 | 4,194,304 | 4.00 |

**Gesamt:** 33,554,432 Bytes = **32.00 MiB**.

### Resonance Field

Standardannahme: N=1,048,576, K=8, C=4; Quantum Statevector nutzt q=24.

| Buffer | Typ | Elemente | Bytes/Element | Bytes | MiB |
|---|---|---|---|---|---|
| state | float | N | 4 | 4,194,304 | 4.00 |
| velocity | float | N | 4 | 4,194,304 | 4.00 |
| drive | float | N | 4 | 4,194,304 | 4.00 |
| energy | float | N | 4 | 4,194,304 | 4.00 |
| neighbors | int | N*K | 4 | 33,554,432 | 32.00 |
| weights | float | N*K | 4 | 33,554,432 | 32.00 |

**Gesamt:** 83,886,080 Bytes = **80.00 MiB**.

### Energy Scheduler

Standardannahme: N=1,048,576, K=8, C=4; Quantum Statevector nutzt q=24.

| Buffer | Typ | Elemente | Bytes/Element | Bytes | MiB |
|---|---|---|---|---|---|
| active_flags | uchar | N | 1 | 1,048,576 | 1.00 |
| active_indices | int | N | 4 | 4,194,304 | 4.00 |
| active_count | uint | 1 | 4 | 4 | 0.00 |

**Gesamt:** 5,242,884 Bytes = **5.00 MiB**.

### Morphogenesis

Standardannahme: N=1,048,576, K=8, C=4; Quantum Statevector nutzt q=24.

| Buffer | Typ | Elemente | Bytes/Element | Bytes | MiB |
|---|---|---|---|---|---|
| cell_type | uchar | N | 1 | 1,048,576 | 1.00 |
| nutrient | float | N | 4 | 4,194,304 | 4.00 |
| energy | float | N | 4 | 4,194,304 | 4.00 |
| potential | float | N | 4 | 4,194,304 | 4.00 |
| rule tables | mixed | R | 20 | 320 | 0.00 |

**Gesamt:** 13,631,808 Bytes = **13.00 MiB**.

### Bio-Brain Standard

Standardannahme: N=1,048,576, K=8, C=4; Quantum Statevector nutzt q=24.

| Buffer | Typ | Elemente | Bytes/Element | Bytes | MiB |
|---|---|---|---|---|---|
| neuron_v | float | N | 4 | 4,194,304 | 4.00 |
| neuron_u | float | N | 4 | 4,194,304 | 4.00 |
| neuron_input | float | N | 4 | 4,194,304 | 4.00 |
| neuron_threshold | float | N | 4 | 4,194,304 | 4.00 |
| neuron_plasticity | float | N | 4 | 4,194,304 | 4.00 |
| spikes | uint8 | N | 1 | 1,048,576 | 1.00 |
| synapse_weight | float | N*K | 4 | 33,554,432 | 32.00 |
| synapse_src | int | N*K | 4 | 33,554,432 | 32.00 |
| synapse_dst | int | N*K | 4 | 33,554,432 | 32.00 |
| synapse_trace | float | N*K | 4 | 33,554,432 | 32.00 |

**Gesamt:** 156,237,824 Bytes = **149.00 MiB**.

### Quantum Statevector

Standardannahme: N=1,048,576, K=8, C=4; Quantum Statevector nutzt q=24.

| Buffer | Typ | Elemente | Bytes/Element | Bytes | MiB |
|---|---|---|---|---|---|
| amplitude_real_imag | float2 | 2^q | 8 | 134,217,728 | 128.00 |
| probability_scratch | float | 2^q | 4 | 67,108,864 | 64.00 |
| phase_scratch | float | 2^q | 4 | 67,108,864 | 64.00 |

**Gesamt:** 268,435,456 Bytes = **256.00 MiB**.


## Ressourcen-Entscheidungsregeln für Agenten

```text
1. Ermittle N, K, C, R und optional q.
2. Berechne required_bytes für alle gleichzeitig aktiven Domänen.
3. Addiere 15 % temporären Overhead:
   planned_bytes = required_bytes * 1.15
4. Vergleiche mit safe_vram_bytes.
5. Wenn planned_bytes > safe_vram_bytes:
      a. reduziere N
      b. reduziere K
      c. reduziere C
      d. deaktiviere optionale Domänen
      e. führe Domänen sequenziell statt gleichzeitig aus
6. Wenn CL_OUT_OF_RESOURCES trotzdem auftritt:
      a. GPU freigeben/shutdown
      b. N halbieren
      c. Wiederholung nur einmal pro Stufe
```

## CL_OUT_OF_RESOURCES-Prävention

| Symptom | Wahrscheinliche Ursache | Präventive Maßnahme |
|---|---|---|
| Fehler bei `clCreateBuffer` | Einzelbuffer zu groß | Buffergröße einzeln prüfen |
| Fehler bei Kernelstart | lokale Ressourcen/Register/Workgroup zu hoch | Workgroup reduzieren oder Kernel splitten |
| Fehler erst nach mehreren Zyklen | Leck oder fehlendes Release | `shutdown_gpu`/Release-Pfade prüfen |
| Fehler nur bei Display-GPU | Windows reserviert VRAM dynamisch | Sicherheitsfaktor auf 0.50 setzen |
| Fehler nur mit Quantum q hoch | Statevector wächst exponentiell | q reduzieren oder separaten Quantum-Lauf nutzen |

## Agenten-Regel

Ein Agent muss vor jeder Allokation den folgenden Guard anwenden:

```pseudo
required = estimate_driver_vram(config)
planned = required * 1.15

if planned > device.global_mem_size * 0.65:
    config = reduce_problem_size(config)
    explain_reduction_to_user()

if any_single_buffer > device.max_mem_alloc_size * 0.80:
    split_buffer_or_reduce_domain()
```

`CL_DEVICE_MAX_MEM_ALLOC_SIZE` ist oft deutlich kleiner als der gesamte VRAM. Ein einzelner großer Pheromon- oder Statevector-Buffer kann daher scheitern, obwohl global noch VRAM frei wirkt.

## Research Extensions: VRAM-Belegung

Die Research-Erweiterungen verwenden kompakte Feld- und Graphbuffer. Die folgenden Tabellen sind bewusst konservativ und enthalten nur die expliziten Datenbuffer, nicht OpenCL-Programme, Kernelhandles, Treiberreserven oder temporäre Runtime-Allokationen.

### Formeln

| Domäne | Buffer | Formel |
|---|---|---:|
| Thermodynamic Langevin | state, momentum, bias, free_energy, neighbors, weights | `N*4*4 + N*K*8` |
| E-I Plastic Reservoir | signal, next_signal, input_drive, gain_e, gain_i, activity_ema, neuron_type, neighbors, weights | `N*4*6 + N*1 + N*K*8` |
| Tensor Bond Entropy Gate | left_factors, right_factors, bond_value, bond_entropy, update_flags, active_bonds, active_count | `B*D*8 + B*13 + 4` |
| Research Trio ohne Tensor | Resonance + Langevin + E-I, gemeinsame Graphen nicht geteilt | ca. `N*(16 + 16 + 25) + 3*N*K*8` |
| Research Trio mit geteilten Nachbarn/Gewichten | Nachbargraph wird einmal gehalten | ca. `N*(16 + 16 + 25) + N*K*8` |

### Standard-Grid 1024 x 1024, K=8

| Domäne | Parameter | Geschätzter VRAM |
|---|---:|---:|
| Thermodynamic Langevin | `N=1,048,576`, `K=8` | 80 MB |
| E-I Plastic Reservoir | `N=1,048,576`, `K=8` | 89 MB |
| Tensor Bond Gate klein | `B=1,048,576`, `D=8` | 77 MB |
| Tensor Bond Gate mittel | `B=1,048,576`, `D=32` | 269 MB |
| Tensor Bond Gate groß | `B=1,048,576`, `D=64` | 525 MB |
| Tensor Bond Gate sehr groß | `B=1,048,576`, `D=128` | 1,037 MB |

### Skalierungstabelle Tensor Bond Gate

| B | D=8 | D=32 | D=64 | D=128 |
|---:|---:|---:|---:|---:|
| 65,536 | 4.8 MB | 16.8 MB | 32.8 MB | 64.8 MB |
| 262,144 | 19.3 MB | 67.3 MB | 131.3 MB | 259.3 MB |
| 1,048,576 | 77.0 MB | 269.0 MB | 525.0 MB | 1,037.0 MB |
| 4,194,304 | 308.0 MB | 1,076.0 MB | 2,100.0 MB | 4,148.0 MB |

### Agentenregeln für Research-Workloads

```pseudo
required = base_domains + research_domains + temp_overhead

if tensor_gate_enabled:
    if D > 64 and B >= 1_000_000:
        require high_vram_profile

if required > global_mem * 0.65:
    reduce N first
    then reduce K
    then reduce B or D
    then execute domains sequentially instead of concurrently
```

### CL_OUT_OF_RESOURCES-Prävention für Research-Algorithmen

| Symptom | Spezifische Ursache | Reduktion |
|---|---|---|
| Fehler beim Tensor-Gate | `B*D*8` dominiert | `D` halbieren |
| Fehler bei Feldalgorithmen | Graphbuffer `N*K*8` dominiert | `K` von 16 auf 8 oder 4 |
| Fehler erst nach mehreren Kerneln | Programme/Runtime-Overhead | Domänen nacheinander initialisieren |
| Fehler nur auf Display-GPU | Windows/GUI belegt VRAM | Safe-Fraction auf 0.50 senken |
