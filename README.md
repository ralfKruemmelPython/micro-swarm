# Micro-Swarm  
**Biologisch inspiriertes, agentenbasiertes Schwarm- und Gedächtnissystem (C++17)**

Micro-Swarm ist ein experimentelles Artificial-Life-System, das untersucht, ob aus **lokalen Regeln** und **mehrschichtigen Gedächtnisformen** globale Struktur, Anpassung und stabile Pfade entstehen können – **ohne klassische neuronale Netze**, ohne Backpropagation und ohne Reinforcement-Learning-Frameworks.

Der Fokus liegt auf:
- Emergenz statt Optimierung
- Kausal nachvollziehbaren Mechanismen
- Trennung von Kurzzeit-, Mittelzeit- und Langzeitgedächtnis

---

## Zentrale Konzepte

### 1. Agenten
Mobile Einheiten mit lokalem Zustand:
- Position, Richtung, Energie
- Genome (Parametervektor)

Agenten handeln ausschließlich lokal und besitzen **keine globale Sicht**.

### 2. Felder (Environment)
Alle Felder sind reguläre 2D-Raster (`GridField`):

- **Ressourcenfeld**  
  Langsam regenerierend, begrenzt (`resource_max`).

- **Pheromonfeld**  
  Diffusiv, verdampfend. Dient der stigmergischen Kommunikation.

- **Molekülfeld (Kurzzeitgedächtnis)**  
  Stark verdampfend, lokal, schnelle Reaktion auf aktuelle Ereignisse.

### 3. Mycel-Netzwerk (Strukturelles Gedächtnis)
Ein langsames Dichtefeld, das **dauerhafte Aktivität** speichert und stabilisiert.

**Aktuelle Dynamik (wichtig):**
- Normalisierter Aktivitäts-Drive aus Pheromon + Ressourcen
- Aktivitätsschwelle (unterhalb kein Wachstum)
- **Logistisches Wachstum** (kein globales Aufpumpen)
- **Diffusionsartiger Transport (Laplacian)** statt positiver Rückkopplung
- Expliziter Decay

→ verhindert globale Sättigung (`mycel_avg != 1.0` Dauerzustand)

### 4. DNA-Gedächtnis (Langzeit)
- Pool erfolgreicher Genome
- Fitness-gewichtetes Sampling
- Mutationen bei Reproduktion
- Alterung (Fitness-Decay)

DNA speichert **Strategien**, nicht Zustände.

---

## Agentenlogik (lokal, pro Schritt)

1. **Sensorik**
   - Drei Richtungen (links / vorne / rechts)
   - Gewichtung aus:
     - Pheromon
     - Ressourcen
     - Molekülen
     - Exploration-Bias

2. **Bewegung**
   - Stochastische Wahl proportional zu lokalen Gewichten
   - Kein globales Ziel

3. **Interaktion**
   - Ressourcenaufnahme
   - Energiegewinn
   - Pheromon- und Molekül-Emission

4. **Selektion**
   - Hohe Energie → Eintrag ins DNA-Gedächtnis
   - Niedrige Energie → Reinitialisierung aus DNA-Pool

---

## Build & Toolchain (Windows / MSVC)

Das Projekt wird **nativ mit Visual Studio 2022** kompiliert und nutzt **CMake ≥ 3.20**.

### Build-Workflow (bewährt)

```powershell
# 1. Hängenden Prozess hart beenden
Get-Process "micro_swarm" -ErrorAction SilentlyContinue | Stop-Process -Force; `
# 2. Kurz warten (Windows gibt Handles frei)
Start-Sleep -Seconds 2; `
# 3. Rebuild
$CMake="C:\Program Files\CMake\bin\cmake.exe"; `
if(Test-Path build){
    Remove-Item -Recurse -Force build -ErrorAction SilentlyContinue
}; `
& $CMake -S . -B build -G "Visual Studio 17 2022" -A x64; `
& $CMake --build build --config Release -j 8
````

---

## Ausführung

```powershell
.\build\Release\micro_swarm.exe
```

---

## CLI-Parameter (vollständig)

### Basis

```
--width N
--height N
--agents N
--steps N
--seed N
```

### Startfelder (CSV)

```
--resources  resources.csv
--pheromone  pheromone.csv
--molecules  molecules.csv
```

CSV-Format:

* Zeilen = Rasterzeilen
* Kommagetrennte Floats
* `#` am Zeilenanfang = Kommentar

---

### Mycel-Tuning (neu)

```
--mycel-growth F
--mycel-decay F
--mycel-transport F
--mycel-threshold F
--mycel-drive-p F
--mycel-drive-r F
```

Diese Parameter erlauben **Live-Tuning**, ohne Recompile.

---

### Feld-Dumps (Diagnose / Analyse)

```
--dump-every N        # 0 = aus
--dump-dir PATH       # Default: dumps
--dump-prefix NAME    # Default: swarm
```

Beispiel:

```powershell
.\micro_swarm.exe --steps 500 --dump-every 50 --dump-dir dumps --dump-prefix test
```

Erzeugt:

```
dumps/
 ├─ test_step000000_resources.csv
 ├─ test_step000000_pheromone.csv
 ├─ test_step000000_molecules.csv
 ├─ test_step000000_mycel.csv
 ├─ test_step000050_...
```

Ideal für:

* Heatmaps
* Zeitraffer
* Emergenz-Analyse

---

## Erwartetes Systemverhalten

* **Mycel** bildet stabile Pfade, aber keine globale Sättigung
* **Pheromone** reagieren schnell, sind flüchtig
* **DNA-Pool** wächst selektiv, nicht explosionsartig
* **Agenten** zeigen:

  * Pfadbildung
  * Lokale Spezialisierung
  * Anpassung bei Umweltänderung

---

## Was Micro-Swarm bewusst nicht ist

* Kein neuronales Netz
* Kein Deep Learning
* Kein Reinforcement-Learning-Framework
* Keine Blackbox-Optimierung

Alle Effekte sind **mechanistisch erklärbar**.

---

## Nächste sinnvolle Experimente

* Ablationstests (Pheromon / Mycel / DNA aus)
* Mehrkanal-Pheromone (z. B. Nahrung vs. Gefahr)
* Unterschiedliche Agentenrollen
* GPU-Beschleunigung der Felddiffusion (OpenCL-Kernel)

---

## Status

**Forschungs- und Experimentalsystem**
Stabil, deterministisch bei festem Seed, vollständig instrumentierbar.

---

**Autor:**
Ralf Krümmel
Artificial Life / Emergent Systems / Low-Level Simulation

