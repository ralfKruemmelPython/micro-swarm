# Micro-Swarm: Biologisch inspiriertes Agenten-Netzwerk

Dieses Projekt implementiert ein experimentelles, agentenbasiertes System, das lokale Regeln aus Mycel-Wachstum, Pheromonkommunikation und evolutionären Prozessen kombiniert. Es nutzt **keine** klassischen neuronalen Netze, keine Backpropagation und keinen Gradientendescent. Der Fokus liegt auf der Frage, ob aus lokalen Interaktionen globale Struktur, Gedächtnisbildung und Anpassungsfähigkeit entstehen.

## Architektur-Ueberblick

Das System ist in klar definierte Subsysteme zerlegt:

1. **Agenten**: Mobile Einheiten mit kurzem Zustand (Energie, Position, Richtung) und einem **Genome** (Parameter).
2. **Umwelt**: Ressourcenfeld mit langsamer Regeneration.
3. **Pheromone**: Diffusives Feld mit Verdampfung, gesteuert durch Agenten-Erfolg.
4. **Molekuele (kurzlebige Zustaende)**: Schnell verdampfendes Feld fuer kurzfristige Kopplung.
5. **Mycel-Netzwerk**: Dichtefeld, das stabile Pfade verstaerkt und Ressourcen-/Pheromonfluesse stabilisiert.
6. **DNA-Gedaechtnis**: Langzeitgedaechtnis als Pool erfolgreicher Genome.

## Datenstrukturen

### Agent
```text
struct Agent {
  Position (x,y), heading, energy
  Genome {sense_gain, pheromone_gain, exploration_bias}
}
```
**Interpretation**:
- `sense_gain`: Sensitivitaet fuer lokale Felder (Reizradius).
- `pheromone_gain`: Gewichtung von Pheromonspuren.
- `exploration_bias`: Zufallseinfluss beim Richtungswechsel.

### Felder (Pheromone, Molekuele, Ressourcen, Mycel)
```text
GridField {
  width, height, data[width * height]
}
```
Jedes Feld ist ein Raster. Diffusion und Verdampfung werden lokal auf 4-Nachbarn angewendet.

### DNA-Gedaechtnis
```text
DNAMemory {
  entries: [ {Genome, fitness, age} ]
}
```
Der Pool speichert erfolgreiche Genome als Langzeitgedaechtnis. Einfache Mutationen erzeugen Varianten.

## Lokale Regeln und Zustandsuebergaenge

### 1) Agentenbewegung (lokale Sensorik)
- Ein Agent prueft drei lokale Sensorpunkte (links, vorne, rechts).
- Gewichtung basiert auf Pheromon, Ressourcen und Molekuelen.
- Der Agent waehlt eine Richtung stochastisch proportional zu den Messwerten.
- Ein Zufallsterm (Exploration) sorgt fuer Diversitaet.

### 2) Ressourcenaufnahme und Pheromon-Emission
- Erntet eine kleine Menge Ressourcen aus der aktuellen Zelle.
- Erhoeht Energie proportional zur Ernte.
- Deponiert Pheromon proportional zum Erfolg.
- Schreibt kurzlebige Molekuele als schnelle, lokale Markierung.

### 3) Pheromone/Molekuele
- Diffusion ueber Nachbarzellen.
- Exponentielle Verdampfung verhindert Saturierung.
- Erlaubt kurzzeitige Orientierung und Pfadbildung.

### 4) Mycel-Netzwerk
- Dichteverstaerkung bei hohem Pheromon- und Ressourcenwert.
- Transportterm koppelt Nachbarn und stabilisiert Pfade.
- Langsame Decay-Rate bewirkt strukturelle Persistenz.

### 5) DNA-Gedaechtnis (Langzeit)
- Erfolgreiche Agenten legen ihr Genome im Pool ab.
- Fitness basiert auf Energieanstieg.
- Neue Agenten werden aus dem Pool gesampelt und leicht mutiert.
- Fitness nimmt langsam ab (Alterung).

### 6) Agentenzyklus
- Energiekosten pro Schritt.
- Bei Energiemangel: Reinitialisierung mit einem Genome aus dem Pool.
- Bei Energiespitzen: Eintrag in die DNA.

## Emergenz: Plausibilitaet und Kontrolle

Die Kombination aus:
- kurzfristigen Signalspuren (Pheromon/Molekuele),
- mittel-langfristiger Struktur (Mycel-Dichte),
- und Langzeitgedaechtnis (DNA-Pool)

erzeugt eine klare Trennschaerfe zwischen kurz- und langfristigen Effekten. Die erwarteten emergenten Effekte sind: stabilisierte Pfade, lokale Spezialisierung und selektive Erinnerung erfolgreicher Strategien.

## OpenCL-Integration (Praxis-Hinweis)

In `src/compute/opencl_loader.cpp` wird zur Laufzeit nach einer vorhandenen OpenCL-Runtime gesucht. Das Projekt nutzt aktuell den CPU-Fallback. Eine direkte Einbettung eines herstellerspezifischen OpenCL-Treibers ist in der Praxis und Lizenzlage nicht sinnvoll moeglich. Der Loader ist so gestaltet, dass eine spaetere GPU-Implementierung minimalen Umbau erfordert.

## Build

```bash
cmake -S . -B build
cmake --build build
```

## Ausfuehrung

```bash
./build/micro_swarm
```

## Verzeichnisstruktur

- `src/main.cpp`: Simulationsloop, Logging, Initialisierung.
- `src/sim/*.h|*.cpp`: Subsysteme und lokale Regeln.
- `src/compute/opencl_loader.*`: Runtime-Probe fuer OpenCL.

## Naechste Erweiterungen

- GPU-Kernel fuer Diffusion und Agenten-Updates.
- Mehrkanalige Pheromone (z. B. Nahrung vs. Gefahr).
- Unterschiedliche Agentenrollen und kooperative Strategien.
- Explizite Mycel-Transportfluesse ueber gerichtete Kanten.
