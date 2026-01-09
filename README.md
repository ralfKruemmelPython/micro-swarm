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

---

## 1) **Baseline / Paper-Run**

```powershell
.\micro_swarm.exe --steps 500 --agents 512 --seed 42 --dump-every 50 --dump-dir dumps --dump-prefix baseline --report-html dumps\baseline_report.html --report-downsample 32 --report-hist-bins 64 --paper-mode --report-global-norm Baseline-Paper_Run
```

**Was das ist:**
Der **Referenzlauf** des Systems ohne externe Störungen und ohne Evolution-Tuning.

**Was er macht:**

* Führt die Simulation deterministisch aus (Seed = 42)
* Schreibt alle 50 Schritte Feld-Dumps (Ressourcen, Pheromone, Moleküle, Mycel)
* Erzeugt einen **wissenschaftlichen HTML-Report**:

  * globale Normalisierung (zeitlich vergleichbare Heatmaps)
  * Entropie, p95, nonzero_ratio
  * Sparklines über die Zeit
  * zusätzlich `baseline_metrics.csv` (Paper-Modus)

**Wofür gedacht:**

* **Ground Truth**
* Vergleichsbasis für Stress- und Evolutionsläufe
* Analyse von Emergenz *ohne* Eingriffe

---

## 2) **Stress-Test / Adaptionslauf**

```powershell
.\micro_swarm.exe --steps 500 --agents 512 --seed 42 --dump-every 50 --dump-dir dumps --dump-prefix stress --report-html dumps\stress_report.html --report-downsample 32 --report-hist-bins 64 --stress-enable --stress-at-step 120 --stress-block-rect 40 40 30 30 --stress-pheromone-noise 0.004 --stress-seed 1337 Stress-Test_Adaptionslauf
```

**Was das ist:**
Ein **Umwelt-Störungstest**, der die Robustheit und Adaptionsfähigkeit des Schwarms prüft.

**Was er macht:**

* Läuft zunächst identisch zur Baseline
* Ab Schritt **120**:

  * blockiert ein Ressourcen-Rechteck (40,40 – 30×30)
  * injiziert kontinuierliches Pheromon-Rauschen
* Alle Effekte werden im Report als *Scenario* dokumentiert

**Wofür gedacht:**

* Prüfen, ob:

  * Mycel-Pfade umlernen
  * Agenten neue Strategien finden
  * das System stabil bleibt (keine Explosion / kein Kollaps)
* **Resilienz-Analyse**

---

## 3) **Evolution / Selektion scharf gestellt**

```powershell
.\micro_swarm.exe --steps 500 --agents 512 --seed 42 --dump-every 50 --dump-dir dumps --dump-prefix evo --report-html dumps\evo_report.html --report-downsample 32 --report-hist-bins 64 --evo-enable --evo-elite-frac 0.20 --evo-min-energy-to-store 1.6 --evo-mutation-sigma 0.05 --evo-exploration-delta 0.05 --evo-fitness-window 50 --evo-age-decay 0.995 Evolution_Selektion_scharf_gestellt ```

**Was das ist:**
Ein **selektiver Evolutionslauf**, bei dem DNA nicht mehr „nebenbei“, sondern gezielt entsteht.

**Was er macht:**

* Aktiviert Elite-Selektion (Top 20 %)
* Speichert Genome nur bei *echtem* Fitness-Gewinn
* Fitness basiert auf **Energiezuwachs über Zeitfenster**, nicht auf Momentwerten
* Mutation ist kontrolliert, keine Drift-Explosion
* Report zeigt:

  * DNA-Pool-Dynamik
  * Energieverteilung
  * Entropie-Änderungen

**Wofür gedacht:**

* Beobachtung echter **Strategie-Evolution**
* Nachweis, dass DNA ein **Langzeitgedächtnis für Verhalten** ist
* Trennung von kurzfristigem Erfolg vs. nachhaltiger Fitness

---

## Kurz gesagt

| Run       | Zweck                         | Frage, die er beantwortet       |
| --------- | ----------------------------- | ------------------------------- |
| Baseline  | Referenz / Paper              | „Was passiert ohne Eingriffe?“  |
| Stress    | Robustheit / Adaptation       | „Kann das System umlernen?“     |
| Evolution | Selektion / Gedächtnisbildung | „Entstehen bessere Strategien?“ |

Wenn du willst, formuliere ich dir daraus direkt einen **README-Abschnitt „Experiment Presets“** oder eine **wissenschaftliche Ergebnisinterpretation** auf Basis der Reports.




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

