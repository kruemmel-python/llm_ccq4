# Enterprise Research Algorithms

Version: Research Extensions  
Stand: 2026-05-06

Dieses Dokument beschreibt die zusätzlichen Enterprise-Research-Algorithmen in `CC_OpenCl.dll`. Es ist für drei Zielgruppen geschrieben:

1. Entwickler, die die DLL direkt aus C/C++ oder Python nutzen.
2. KI-Agenten, die anhand dokumentierter Verträge sichere Pipeline-Entscheidungen treffen.
3. Systemintegratoren, die VRAM, Stabilität, Fehlermodi und Capability-Fallbacks planen müssen.

Die Research-Erweiterung ergänzt den bestehenden Enterprise Algorithm Pack um vier GPU-Domänen:

| Domäne | Exportfunktion | Kernel | Primärer Nutzen |
|---|---|---|---|
| Thermodynamic Langevin Field | `execute_thermodynamic_langevin_step_gpu` | `cc_thermodynamic_langevin_step` | stochastische Relaxation, Energie-Minimierung, probabilistische Feldsuche |
| E-I Plastic Reservoir | `execute_ei_plastic_reservoir_step_gpu` | `cc_ei_plastic_reservoir_step` | excitatory/inhibitory Dynamik, Reservoir Computing, adaptive Agentensignale |
| Tensor Bond Entropy Gate | `execute_tensor_bond_entropy_gate_gpu` | `cc_tensor_bond_entropy_gate` | Tensor-Bond-Bewertung, Entropie-Gating, aktive Arbeitslisten |
| Deep Substrate Dispatch | `execute_deep_substrate_dispatch_gpu` | `cc_deep_substrate_dispatch` | LLM-Schicht-Routing auf Dense/Mycel/Tensor/Reservoir/Langevin/SubQG-Pfade |

Die bereits vorhandene Resonance-Domäne bleibt die bevorzugte Basisschicht für gekoppelte Feldpropagation.

---

## 0. Deep Substrate Dispatch

### Zweck

`execute_deep_substrate_dispatch_gpu` ist die Brücke zwischen LLM-Schichten und den spezialisierten Substrat-Kerneln. Der Kernel entscheidet pro Layer, welcher Pfad als nächstes ausgeführt werden soll:

| Route | Bedeutung |
|---:|---|
| `0` | Dense/GEMM-Fallback |
| `1` | Mycel-Sparse-Routing |
| `2` | Tensor-Bond-Entropy-Gate |
| `3` | E/I Plastic Reservoir |
| `4` | Thermodynamic Langevin |
| `5` | SubQG/Quantum-Coupling |

### API

```c
int execute_deep_substrate_dispatch_gpu(
    int gpu_index,
    void* layer_role_buf,
    void* layer_priority_buf,
    void* emotion_state_buf,
    void* nutrient_state_buf,
    void* route_scores_buf,
    void* selected_route_buf,
    void* dispatch_params_buf,
    void* active_layers_buf,
    void* active_count_buf,
    int layer_count,
    int route_count,
    float relevance_threshold,
    float exploration_bias
);
```

### Buffervertrag

| Buffer | Typ | Elemente | Richtung | Bedeutung |
|---|---:|---:|---|---|
| `layer_role_buf` | `int` | `L` | in | `0` dense/logits, `1` attention, `2` MLP, `3` norm/residual, `4` sparse router |
| `layer_priority_buf` | `float` | `L` | in | Relevanz pro Layer |
| `emotion_state_buf` | `float` | `4` | in | precision, novelty, dream, risk |
| `nutrient_state_buf` | `float` | `L` | in | Mycel-/Aktivitätsbudget pro Layer |
| `route_scores_buf` | `float` | `L*route_count` | out | Score pro Route |
| `selected_route_buf` | `int` | `L` | out | Gewählte Route |
| `dispatch_params_buf` | `float` | `L*4` | out | precision_scale, exploration_scale, risk_scale, relevance |
| `active_layers_buf` | `int` | `L` | out | Kompaktierte aktive Layer |
| `active_count_buf` | `uint` | `1` | out | Anzahl aktiver Layer |

`route_count` muss mindestens `6` sein. Der Host nutzt `selected_route` und `dispatch_params`, um anschließend die passenden bestehenden Kernel-Familien auszuführen.

### Gemma-Metadaten vorbereiten

Für `google/gemma-3n-E4B` erzeugt der Offline-Inspector ein Dispatcher-Manifest aus `config.json` und `model.safetensors.index.json`:

```powershell
python .\scripts\gemma_substrate_inspector.py `
  --model-dir D:\models\gemma-3n-E4B `
  --out build\gemma_substrate_manifest.json `
  --c-arrays-out build\gemma_substrate_arrays.h
```

Wenn `huggingface_hub` installiert ist und die Gemma-Lizenz im HF-Account akzeptiert wurde, kann der Inspector die Metadaten auch direkt laden:

```powershell
$env:HF_TOKEN="hf_..."
python .\scripts\gemma_substrate_inspector.py --hf-model google/gemma-3n-E4B
```

Sobald die Safetensors-Shards lokal liegen, prueft der Weight-Index einzelne Layer und Tensoren ohne Voll-Load:

```powershell
$env:PYTHONPATH="subqg_driver_tests"
python -m gemma_runtime.weights `
  --model-dir D:\models\gemma-3n-E4B `
  --layer 0 `
  --require-shards
```

Ein einzelner Tensor kann bytegenau gelesen werden:

```powershell
python -m gemma_runtime.weights `
  --model-dir D:\models\gemma-3n-E4B `
  --tensor language_model.model.layers.0.self_attn.q_proj.weight `
  --read-bytes
```

Der erste blockweise Quantizer erzeugt ein einfaches `CCQ4`-Format mit `float32`-Skalen pro Block und gepackten signed INT4-Werten:

```powershell
$env:PYTHONPATH="subqg_driver_tests"
python -m gemma_runtime.quantizer `
  --model-dir D:\models\gemma-3n-E4B `
  --tensor model.language_model.layers.0.self_attn.k_proj.weight `
  --output-dir build\gemma_quantized `
  --manifest-out build\gemma_quantized\layer0_k_proj_manifest.json
```

Vor einem groesseren Lauf kann die Tensor-Auswahl trocken geprueft werden:

```powershell
python -m gemma_runtime.quantizer `
  --model-dir D:\models\gemma-3n-E4B `
  --layer 0 `
  --group attention `
  --dry-run
```

Ein erzeugter `CCQ4`-Tensor kann ohne Modelldateien inspiziert und blockweise dequantisiert werden:

```powershell
python -m gemma_runtime.quantizer `
  --inspect-ccq4 build\gemma_quantized\model.language_model.layers.0.self_attn.k_proj.weight.ccq4

python -m gemma_runtime.quantizer `
  --dequantize-ccq4 build\gemma_quantized\model.language_model.layers.0.self_attn.k_proj.weight.ccq4 `
  --start-block 0 `
  --block-count 2
```

Der CPU-Referenzpfad fuer CCQ4-MatVec validiert das spaetere OpenCL-Dequant-Matmul-Verhalten:

```powershell
python -m gemma_runtime.matvec `
  --ccq4 build\gemma_quantized\model.language_model.layers.0.self_attn.k_proj.weight.ccq4 `
  --json-out build\gemma_quantized\layer0_k_proj_matvec.json `
  --output-f32 build\gemma_quantized\layer0_k_proj_matvec.f32
```

Der passende OpenCL-Pfad ist als `execute_ccq4_matvec_gpu` exportiert. Buffervertrag:

| Buffer | Typ | Elemente | Bedeutung |
|---|---:|---:|---|
| `packed_weights_buf` | `uchar` | `rows * ceil(cols/block_size) * ceil(block_size/2)` | signed INT4, low nibble zuerst |
| `scales_buf` | `float` | `rows * ceil(cols/block_size)` | ein Scale pro Block |
| `input_vec_buf` | `float` | `cols` | Aktivierungsvektor |
| `output_vec_buf` | `float` | `rows` | Ergebnis |

Smoke-Test:

```powershell
$env:PYTHONPATH="subqg_driver_tests"
python .\subqg_driver_tests\test_47_ccq4_matvec_gpu.py --dll .\build\CC_OpenCl.dll --gpu 0
```

Eine echte `CCQ4`-Datei kann ueber die Runtime-Bridge geladen, auf GPU-Buffer geschrieben und gegen die CPU-Referenz verglichen werden:

```powershell
python -m gemma_runtime.matvec `
  --ccq4 build\gemma_quantized\model.language_model.layers.0.self_attn.k_proj.weight.ccq4 `
  --gpu-dll build\CC_OpenCl.dll `
  --compare-gpu `
  --json-out build\gemma_quantized\layer0_k_proj_matvec_gpu_compare.json
```

Layer-0 Q/K/V-Projektionen koennen als Gruppe quantisiert und ueber GPU-CCQ4 ausgefuehrt werden:

```powershell
python -m gemma_runtime.projection `
  --model-dir D:\models\gemma-3n-E4B `
  --layer 0 `
  --output-dir build\gemma_quantized `
  --dll build\CC_OpenCl.dll `
  --compare-cpu `
  --json-out build\gemma_quantized\layer0_attention_projection_runtime.json
```

Die Runtime haelt dabei eine `GpuCcq4Session` offen und fuehrt mehrere Projektionen in derselben Treiber-Initialisierung aus. Das vermeidet wiederholtes OpenCL-Setup fuer Q/K/V/O und ist der Host-seitige Pfad fuer die naechste Multi-Layer-Ausbaustufe.

Der minimale Attention-Core baut daraus einen Single-Token-Kontext und projiziert diesen wieder durch `o_proj`:

```powershell
python -m gemma_runtime.projection `
  --model-dir D:\models\gemma-3n-E4B `
  --layer 0 `
  --output-dir build\gemma_quantized `
  --dll build\CC_OpenCl.dll `
  --compare-cpu `
  --attention-core `
  --json-out build\gemma_quantized\layer0_attention_core_runtime.json
```

Ein vollstaendiges CCQ4-Artefakt fuer den Gemma-3n Language-Teil wird in einem resumierbaren Lauf erzeugt. Der Pfad streamt Safetensors blockweise und laedt auch sehr grosse Tensoren nicht komplett in Python-Listen:

```powershell
$env:PYTHONPATH="subqg_driver_tests"
python -m gemma_runtime.build_full_model `
  --model-dir D:\models\gemma-3n-E4B `
  --output-dir build\gemma_full_ccq4 `
  --include-embeddings `
  --progress-every 25
```

Das Ergebnis ist `build\gemma_full_ccq4\ccq4_full_language_manifest.json` plus eine `.ccq4`-Datei pro Language-Tensor. Dieser Ordner kann direkt als `--output-dir` fuer die Projektionsruntime genutzt werden; vorhandene `.ccq4`-Dateien werden wiederverwendet.

Der erste Token-Forward-Loop nutzt dieses Paket fuer Embedding-Lookup, RMSNorm, RoPE, KV-Cache-Attention, MLP/GELU, Residuals, final Norm und tied-Embedding-Logits:

```powershell
$env:PYTHONPATH="subqg_driver_tests"
python -m gemma_runtime.forward `
  --ccq4-dir build\gemma_full_ccq4 `
  --dll build\CC_OpenCl.dll `
  --token-id 2 `
  --position 0 `
  --max-layers 1 `
  --top-k 8 `
  --vocab-limit 2048 `
  --json-out build\gemma_full_ccq4\forward_token2_layer1.json
```

Fuer Sequenzen bleiben die KV-Caches im Runtime-Objekt erhalten:

```powershell
python -m gemma_runtime.forward `
  --ccq4-dir build\gemma_full_ccq4 `
  --dll build\CC_OpenCl.dll `
  --tokens 2,186 `
  --max-layers 1 `
  --skip-mlp `
  --top-k 5 `
  --vocab-limit 512
```

Standardmaessig nutzt der Forward-Loop persistente GPU-Gewichte: jede CCQ4-Matrix wird pro `GpuCcq4Session` einmal als `packed`/`scales`-Buffer hochgeladen und danach wiederverwendet. Neue DLLs besitzen dafuer eine Treiber-Registry:

- `cc_register_persistent_ccq4_weight`
- `cc_execute_resident_ccq4_matvec`
- `cc_release_persistent_weight`
- `cc_release_all_persistent_weights`
- `cc_get_persistent_weight_count`

Zum Vergleich kann Residency abgeschaltet werden:

```powershell
python -m gemma_runtime.forward `
  --ccq4-dir build\gemma_full_ccq4 `
  --dll build\CC_OpenCl.dll `
  --tokens 2,186 `
  --max-layers 1 `
  --skip-mlp `
  --no-resident-weights
```

Das prompt-testbare Enterprise-Modell kapselt Tokenizer, Prompt-Prefill, KV-Cache, Greedy-Decoding und Text-Decode:

```powershell
$env:PYTHONPATH="subqg_driver_tests"
python -m gemma_runtime.enterprise_model `
  --prompt "Hello enterprise" `
  --ccq4-dir build\gemma_full_ccq4 `
  --dll build\CC_OpenCl.dll `
  --max-layers 1 `
  --max-new-tokens 1 `
  --top-k 8 `
  --vocab-limit 2048 `
  --json-out build\gemma_full_ccq4\enterprise_prompt_hello_layer1.json
```

`--prompt` akzeptiert auch eine JSON-Liste von Token-IDs, zum Beispiel `--prompt "[2,9259,18315]"`. Fuer einen volleren Lauf kann `--max-layers` schrittweise bis `35` angehoben werden; auf 4GB-GPUs sollte dabei zuerst mit kleinem Prompt und begrenztem `--vocab-limit` getestet werden.

Fuer autonome Sessions kann ein System-Prompt plus emotionsgesteuertes Sampling gesetzt werden:

```powershell
python -m gemma_runtime.enterprise_model `
  --system-prompt "Du bist das Sovereign-CC-Modell. Antworte knapp und technisch." `
  --prompt "Hello enterprise" `
  --ccq4-dir build\gemma_full_ccq4 `
  --dll build\CC_OpenCl.dll `
  --max-layers 1 `
  --max-new-tokens 1 `
  --emotion-mode precise `
  --temperature 0 `
  --top-k 8 `
  --vocab-limit 1024
```

`--emotion-mode` steuert Default-Werte fuer `temperature`, `top_p` und `repetition_penalty`: `precise`, `calm`, `creative`, `dream`. Explizite CLI-Werte ueberschreiben diese Defaults.
Standardmaessig verhindert `--no-repeat-ngram-size 2`, dass bereits erzeugte Bigramme wiederholt werden. Fuer Diagnose-Laeufe kann der Wert auf `0` gesetzt werden.

Der interaktive Session-Manager haelt System-Prompt, KV-Cache, GPU-Session und residente CCQ4-Matrizen ueber mehrere User-Turns offen:

```powershell
$env:PYTHONPATH="subqg_driver_tests"
python -m gemma_runtime.enterprise_model `
  --interactive `
  --system-prompt "Du bist das Sovereign-CC-Modell. Antworte knapp und technisch." `
  --ccq4-dir build\gemma_full_ccq4 `
  --dll build\CC_OpenCl.dll `
  --max-layers 1 `
  --max-new-tokens 8 `
  --emotion-mode precise `
  --no-repeat-ngram-size 2 `
  --top-k 32 `
  --vocab-limit 4096
```

Im interaktiven Modus beendet eine leere Eingabe oder `/exit` die Session. Special-Tokens wie `<bos>`, `<unk>`, `<pad>`, `<mask>` und ungenutzte Platzhalter werden beim Sampling unterdrueckt; `<eos>` bleibt erlaubt, damit die autoregressive Schleife sauber stoppen kann.

Optional kann `--quiet-driver --driver-log build\gemma_full_ccq4\enterprise_driver.log` gesetzt werden. Der Python-Launcher versucht dann, die sehr umfangreichen DLL-/Kernel-Ausgaben in die Logdatei umzuleiten. Auf Windows koennen einzelne native CRT-Flushes trotzdem noch auf der Konsole erscheinen; der Modus reduziert aber den interaktiven Rauschpegel deutlich.

GPU-Auswahl:

- `--gpu 1` startet die Hauptinferenz direkt auf GPU 1.
- `--also-gpu 1` initialisiert GPU 1 zusaetzlich vor der Hauptsession, registriert eine reale CCQ4-Matrix resident und fuehrt einen MatVec-Probe aus. Danach startet die Hauptsession auf `--gpu`.

Beispiel mit Hauptsession auf GPU 0 plus Zusatzprobe auf GPU 1:

```powershell
python -m gemma_runtime.enterprise_model `
  --interactive `
  --also-gpu 1 `
  --system-prompt "Du bist das Sovereign-CC-Modell. Antworte knapp und technisch." `
  --ccq4-dir build\gemma_full_ccq4 `
  --dll build\CC_OpenCl.dll `
  --gpu 0 `
  --max-layers 1 `
  --max-new-tokens 8 `
  --emotion-mode precise `
  --top-k 32 `
  --vocab-limit 4096
```

---

## 1. Architekturübersicht

Die Research-Kernel liegen im Enterprise Algorithm Pack und werden lazy kompiliert. Ein Hostprogramm muss nicht separat Kerneldateien laden. Die DLL verwaltet:

- OpenCL-Plattform- und Geräteauswahl
- Context und Command Queue
- Kernel- und Program-Handles
- Profiling-Ausgabe
- Last-Error-Diagnose
- Cleanup über `shutdown_gpu`

Typischer Ablauf:

```text
initialize_gpu(gpu)
allocate_gpu_memory(...)
write_host_to_gpu_blocking(...)
execute_*_gpu(...)
finish_gpu(gpu)
read_gpu_to_host_blocking(...)
free_gpu_memory(...)
shutdown_gpu(gpu)
```

Wichtig: `void*`-Parameter in der API sind OpenCL-Bufferhandles, die über `allocate_gpu_memory` erzeugt werden.

---

## 2. Thermodynamic Langevin Relaxation Field

### Zweck

`execute_thermodynamic_langevin_step_gpu` führt einen stochastischen Relaxationsschritt auf einem gekoppelten Feld aus. Jeder Knoten besitzt:

- Zustand `state[i]`
- Momentum `momentum[i]`
- Ziel-/Biaswert `bias[i]`
- Ausgabe `free_energy[i]`
- Nachbarschaftskräfte über `neighbors[i*K + j]` und `weights[i*K + j]`

Rauschen wird nicht als Fehler behandelt, sondern als produktiver Suchmechanismus. Der Kernel eignet sich für Sampling, robuste Energie-Minimierung und explorative Agentenfelder.

### API

```c
int execute_thermodynamic_langevin_step_gpu(
    int gpu_index,
    void* state_buf,
    void* momentum_buf,
    void* bias_buf,
    void* free_energy_buf,
    void* neighbors_buf,
    void* weights_buf,
    int N,
    int K,
    float dt,
    float temperature,
    float quartic_alpha,
    float bias_stiffness,
    float coupling,
    float friction,
    unsigned int seed,
    float clamp_abs
);
```

### Buffervertrag

| Buffer | Typ | Elemente | Richtung | Bedeutung |
|---|---:|---:|---|---|
| `state_buf` | `float` | `N` | in/out | Feldzustand |
| `momentum_buf` | `float` | `N` | in/out | dynamische Trägheit |
| `bias_buf` | `float` | `N` | in | Ziel-/Attraktorwert |
| `free_energy_buf` | `float` | `N` | out | lokale Energie-Diagnose |
| `neighbors_buf` | `int` | `N*K` | in | Nachbarschaftsindizes |
| `weights_buf` | `float` | `N*K` | in | Kopplungsgewichte |

### Parametervertrag

| Parameter | Empfohlener Bereich | Wirkung | Gefahr |
|---|---:|---|---|
| `N` | `>= 10_000` | Knotenanzahl | kleine N: Launch-Overhead dominiert |
| `K` | `4`, `8`, `16` | Nachbarn pro Knoten | hohe K: Speicherbandbreite |
| `dt` | `0.001–0.05` | Integrationsschritt | Explosion bei zu großem Wert |
| `temperature` | `0.0–0.10` | Rauschamplitude | Diffusion statt Konvergenz |
| `quartic_alpha` | `0.01–1.0` | quartisches Potential | zu hoch: harte Rückstellkräfte |
| `bias_stiffness` | `0.01–1.0` | Bindung an Bias | zu hoch: Überdämpfung |
| `coupling` | `0.0–1.0` | Nachbarschaftskraft | Oszillation/NaN |
| `friction` | `0.0–0.5` | Momentum-Dämpfung | zu hoch: Feld friert ein |
| `seed` | beliebig | deterministisches Rauschen | gleicher Seed = reproduzierbar |
| `clamp_abs` | `> 0` empfohlen | Stabilitätslimit | `0` erlaubt Explosion |

### Seiteneffekte

- `state_buf` wird überschrieben.
- `momentum_buf` wird überschrieben.
- `free_energy_buf` wird überschrieben.
- `bias_buf`, `neighbors_buf`, `weights_buf` bleiben unverändert.

### Agentenregeln

```pseudo
if output contains NaN or Inf:
    temperature *= 0.5
    coupling *= 0.5
    dt *= 0.5
    set clamp_abs > 0

if free_energy increases for many steps:
    increase friction
    reduce temperature

if field converges too early:
    increase temperature slightly
    reduce bias_stiffness
```

---

## 3. E-I Plastic Reservoir Field

### Zweck

`execute_ei_plastic_reservoir_step_gpu` implementiert ein excitatory/inhibitory Reservoir mit lokaler plastischer Gain-Regelung. Es ist eine GPU-nahe Alternative zu objektbasierten Agenten- oder Neuronenmodellen.

Der Kernel trennt Nachbarschaftsbeiträge nach Neuronentyp:

- `neuron_type[nb] == 1`: excitatory
- `neuron_type[nb] == 0`: inhibitory

Dann wird ein aktivierter Signalwert berechnet und die lokalen Gains `gain_e` und `gain_i` werden in Richtung einer Zielaktivität angepasst.

### API

```c
int execute_ei_plastic_reservoir_step_gpu(
    int gpu_index,
    void* signal_buf,
    void* next_signal_buf,
    void* input_drive_buf,
    void* neighbors_buf,
    void* weights_buf,
    void* neuron_type_buf,
    void* gain_e_buf,
    void* gain_i_buf,
    void* activity_ema_buf,
    int N,
    int K,
    float leak,
    float target_activity,
    float plasticity_rate,
    float input_scale,
    float clamp_gain
);
```

### Buffervertrag

| Buffer | Typ | Elemente | Richtung | Bedeutung |
|---|---:|---:|---|---|
| `signal_buf` | `float` | `N` | in | aktueller Zustand |
| `next_signal_buf` | `float` | `N` | out | nächster Zustand |
| `input_drive_buf` | `float` | `N` | in | externer Stimulus |
| `neighbors_buf` | `int` | `N*K` | in | Nachbarschaftsindizes |
| `weights_buf` | `float` | `N*K` | in | gewichtete Kopplung |
| `neuron_type_buf` | `uchar` | `N` | in | 1 = excitatory, 0 = inhibitory |
| `gain_e_buf` | `float` | `N` | in/out | excitatory Gain |
| `gain_i_buf` | `float` | `N` | in/out | inhibitory Gain |
| `activity_ema_buf` | `float` | `N` | in/out | gleitender Aktivitätsmittelwert |

### Parametervertrag

| Parameter | Empfohlener Bereich | Wirkung | Gefahr |
|---|---:|---|---|
| `leak` | `0.01–0.30` | Mischrate zum neuen Zustand | zu hoch: flackerndes Reservoir |
| `target_activity` | `0.05–0.40` | Zielaktivität | falsch: Gains driften |
| `plasticity_rate` | `0.001–0.05` | Gain-Anpassung | zu hoch: Oszillation |
| `input_scale` | `0.0–2.0` | Stärke externer Inputs | Sättigung bei zu hoch |
| `clamp_gain` | `1.0–10.0` | Gain-Obergrenze | `0` blockiert Dynamik |

### Seiteneffekte

- `next_signal_buf` wird geschrieben.
- `gain_e_buf`, `gain_i_buf`, `activity_ema_buf` werden aktualisiert.
- `signal_buf` wird nicht überschrieben. Für iterative Simulation muss der Host `next_signal` als neues `signal` verwenden oder Buffer tauschen.

### Agentenregeln

```pseudo
if max(gain_e) == clamp_gain or max(gain_i) == clamp_gain:
    reduce plasticity_rate
    increase clamp_gain only if numerically stable

if reservoir output is almost zero:
    increase input_scale
    reduce inhibitory weights
    reduce target_activity only carefully

if output saturates:
    reduce input_scale
    reduce leak
    reduce excitatory weights
```

---

## 4. Tensor Bond Entropy Gate

### Zweck

`execute_tensor_bond_entropy_gate_gpu` bewertet Paare von Faktoren als Tensor-Bonds. Der Kernel berechnet:

1. Bond-Score als Dot Product.
2. Sigmoid-Wahrscheinlichkeit.
3. binäre Entropie.
4. geglätteten Bond-Wert.
5. Update-Flag anhand Entropie oder Residual.
6. aktive Bond-Liste per Atomic Counter.

Der Algorithmus ist für Agenten besonders wertvoll, weil er eine GPU-seitige Arbeitsliste erzeugt. Statt alle Bonds weiterzuverarbeiten, können nur unsichere oder stark veränderte Bonds in nachfolgenden Schritten behandelt werden.

### API

```c
int execute_tensor_bond_entropy_gate_gpu(
    int gpu_index,
    void* left_factors_buf,
    void* right_factors_buf,
    void* bond_value_buf,
    void* bond_entropy_buf,
    void* update_flags_buf,
    void* active_bonds_buf,
    void* active_count_buf,
    int B,
    int D,
    float entropy_threshold,
    float residual_threshold,
    float smoothing
);
```

### Buffervertrag

| Buffer | Typ | Elemente | Richtung | Bedeutung |
|---|---:|---:|---|---|
| `left_factors_buf` | `float` | `B*D` | in | linker Faktor pro Bond |
| `right_factors_buf` | `float` | `B*D` | in | rechter Faktor pro Bond |
| `bond_value_buf` | `float` | `B` | in/out | geglätteter Bondwert |
| `bond_entropy_buf` | `float` | `B` | out | binäre Entropie |
| `update_flags_buf` | `uchar` | `B` | out | 0/1 Update-Flag |
| `active_bonds_buf` | `int` | `B` | out | kompakte aktive Bond-Liste |
| `active_count_buf` | `uint` | `1` | in/out | muss vor Aufruf 0 sein |

### Parametervertrag

| Parameter | Empfohlener Bereich | Wirkung | Gefahr |
|---|---:|---|---|
| `B` | `>= 1024` | Anzahl Bonds | kleine B: Launch-Overhead |
| `D` | `4–256` | Faktordimension | hohe D: Bandbreite/ALU |
| `entropy_threshold` | `0.1–0.69` | Unsicherheits-Gate | zu niedrig: alles aktiv |
| `residual_threshold` | `0.001–0.1` | Änderungs-Gate | zu niedrig: alles aktiv |
| `smoothing` | `0.0–0.99` | Bondwert-Glättung | zu hoch: träge Reaktion |

### Kritischer Vertrag: `active_count`

`active_count_buf` muss vor jedem Aufruf auf `0` gesetzt werden. Andernfalls hängt die neue aktive Liste an alte Werte an.

```pseudo
write active_count[0] = 0
execute_tensor_bond_entropy_gate_gpu(...)
read active_count[0]
```

### Agentenregeln

```pseudo
active_rate = active_count / B

if active_rate > 0.8:
    entropy_threshold *= 1.1
    residual_threshold *= 1.2

if active_rate < 0.001:
    entropy_threshold *= 0.9
    residual_threshold *= 0.8
```

---

## 5. Capability-Fallbacks auf AMD OpenCL

Die beobachtete Laufzeitausgabe kann melden:

```text
Device supports enqueue, but OpenCL 2.0 symbols missing in driver.
```

Das ist kein Fehler. Es bedeutet:

- Das Gerät oder die Header melden OpenCL-2.x-Fähigkeiten.
- Die Runtime stellt jedoch nicht alle 2.x-Symbole direkt als Link-/Dispatch-Ziele bereit.
- Der Treiber verwendet dann klassische Host-Enqueue-Pfade.

Agentenregel:

```pseudo
if device_enqueue_supported and symbols_missing:
    do not fail
    use host-enqueue path
    avoid documentation claims requiring device-side enqueue
```

---

## 6. Research-VRAM-Formeln

| Domäne | Formel | Beispiel |
|---|---:|---:|
| Thermodynamic Langevin | `N*4*4 + N*K*8` | state, momentum, bias, free_energy, neighbors, weights |
| E-I Plastic Reservoir | `N*4*5 + N*1 + N*K*8` | signal, next, drive, gains, ema, types, graph |
| Tensor Bond Entropy Gate | `B*D*8 + B*(4+4+1+4) + 4` | left/right factors, value, entropy, flags, active list, count |

Für `N=1,048,576`, `K=8`:

| Domäne | Geschätzter VRAM |
|---|---:|
| Thermodynamic Langevin | ca. 80 MB |
| E-I Plastic Reservoir | ca. 85 MB |
| Resonance Field | ca. 80 MB |
| Tensor Bond Gate mit `B=1,048,576`, `D=8` | ca. 77 MB |
| Tensor Bond Gate mit `B=1,048,576`, `D=64` | ca. 525 MB |

---

## 7. Beispielhost

Das mitgelieferte Script `docs/examples/enterprise_research_host.py` demonstriert die korrekte Reihenfolge:

1. DLL laden.
2. ctypes-Signaturen setzen.
3. GPU initialisieren.
4. Buffer allozieren.
5. Daten schreiben.
6. Research-Kernel ausführen.
7. Ergebnisse lesen.
8. Finite-Werte prüfen.
9. Ressourcen freigeben.

Ausführung aus der Projektwurzel:

```powershell
python .\enterprise_research_host.py
```

oder:

```powershell
python .\docs\examples\enterprise_research_host.py
```

Erwarteter Erfolg:

```text
ALL ENTERPRISE ALGORITHM TESTS PASSED
```

---

## 8. Wann welcher Research-Algorithmus?

| Ziel | Algorithmus | Warum |
|---|---|---|
| Feld soll explorativ relaxieren | Thermodynamic Langevin | Rauschen + Energiepotential direkt auf GPU |
| Agentensignale sollen stabil, aber adaptiv sein | E-I Plastic Reservoir | lokale E/I-Trennung und Homeostase |
| Nur relevante Tensorbindungen sollen weiterverarbeitet werden | Tensor Bond Entropy Gate | aktive Liste auf GPU |
| Diffuse Aktivität ohne Rauschen | Resonance Field | deterministischer, stabiler Feldmotor |
| Nur aktive Knoten weiterverarbeiten | Energy Scheduler | Aktivitätskompaktierung |

---

## 9. Bekannte Scheitermodi

| Symptom | Wahrscheinliche Ursache | Reaktion |
|---|---|---|
| `CL_OUT_OF_RESOURCES` | N, K, B oder D zu groß | VRAM-Tabelle anwenden, Dimension halbieren |
| NaN in `state` | zu hohe Temperatur/Kopplung/Zeitschritt | clamp setzen, dt halbieren |
| Gains bei `clamp_gain` | zu hohe Plasticity oder falsches Target | Rate senken, Target prüfen |
| `active_count == B` | Gates zu niedrig | Thresholds erhöhen |
| `active_count == 0` | Gates zu hoch oder Faktoren stabil | Thresholds senken |
| Output unverändert | `signal` und `next_signal` nicht getauscht | Double-Buffer korrekt tauschen |
| DLL nicht gefunden | falscher Pfad | `build\CC_OpenCl.dll` prüfen |

---

## 10. Enterprise-Einsatzempfehlung

Für produktive Agentensysteme sollten die Research-Algorithmen nicht isoliert, sondern als Pipeline genutzt werden:

```text
Resonance Field
    -> Energy Scheduler
    -> E-I Plastic Reservoir
    -> Tensor Bond Entropy Gate
    -> optional Langevin Relaxation für unsichere Regionen
```

Die wichtigste Regel: Dimensionen und Arbeitslisten auf der GPU halten. Host-Roundtrips nur für Diagnose, Visualisierung oder finale Ausgabe verwenden.
