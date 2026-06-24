# COS 710 – Assignment 2: Structure-Based Genetic Programming for Regression

**Student:** Yohali Malaika Kamangu | **Number:** u23618583

This project implements **Structure-Based Genetic Programming (SBGP)** for symbolic
regression of UK residential electricity load. It extends the canonical GP from
Assignment 1 by incorporating a **structural fitness sharing** mechanism derived from
the SBGP lecture slides: "counting the number of nodes that are the same from the root."

---

## Project Structure

```
Assignment 2/
├── src/
│   ├── Node.java               — GP tree node + prefixMatchCount (SBGP)
│   ├── Tree.java               — GP tree wrapper + structuralSimilarity (SBGP)
│   ├── DatasetLoader.java      — CSV loader and sliding-window preprocessor
│   ├── GeneticProgramming.java — SBGP engine (fitness sharing evolution loop)
│   └── Main.java               — Entry point + comparison output
├── out/                        — Compiled .class files
├── documentation/
│   └── Residential_Energy_Dataset_UK- 2014-2020.csv
├── GP_Assignment2.jar          — Executable JAR
├── manifest.txt
├── u23618583_YM_Kamangu_710_Assignment_2_Report.pdf — Assignment report
├── results.jpeg                — Terminal output screenshot (10 runs)
├── results_output.txt          — Full stdout from 10-run experiment
└── README.md                   — This file
```

---

## Requirements

- **Java JDK 17+** — No external libraries. Download:
  [https://adoptium.net/temurin/releases/?version=17](https://adoptium.net/temurin/releases/?version=17)
- Dataset CSV must be at `documentation/Residential_Energy_Dataset_UK- 2014-2020.csv`
  relative to the working directory (already present in the `documentation/` folder).

---

## Building

From the `Assignment 2/` directory:

```bash
javac -d out src/*.java
```

Or build the JAR:

```bash
jar cfm GP_Assignment2.jar manifest.txt -C out .
```

---

## Running

**From compiled classes:**
```bash
java -cp out Main
```

**From the JAR:**
```bash
java -jar GP_Assignment2.jar
```

Run from the `Assignment 2/` directory so the dataset path resolves correctly.

---

## SBGP Parameters

| Parameter | Value | Description |
|---|---|---|
| Similarity threshold (σ) | 0.60 | Trees sharing ≥60% prefix structure are neighbours |
| Sharing coefficient (α) | 0.01 | Scales the structural fitness penalty |
| Structural sample size | 50 | Peers sampled per individual for niche count |

**Base GP parameters** (identical to Assignment 1 for fair comparison):

| Parameter | Value |
|---|---|
| Population size | 1,000 |
| Max tree depth | 6 |
| Generations | 100 |
| Tournament size | 2 |
| Crossover rate | 0.90 |
| Mutation rate | 0.10 |
| Elitism | 2% (20 individuals) |
| Window size | 7 |
| Data sample | 100,000 rows |

---

## Quick Start (JAR)

```bash
# From the Assignment 2/ directory:
java -jar GP_Assignment2.jar
```

The JAR is self-contained. Ensure `documentation/Residential_Energy_Dataset_UK- 2014-2020.csv` is present relative to the working directory.  
Expected runtime: ~11 minutes per 10-run experiment.

---

## Experimental Results (10 Runs)

| Run | Seed | Train MSE | Test MSE | Time (s) |
|-----|------|-----------|----------|----------|
| 1 | 101 | 0.00014780 | 0.00017864 | 662.14 |
| 2 | 202 | 0.00014425 | 0.00017096 | 541.28 |
| 3 | 303 | 0.00014899 | 0.00017794 | 780.12 |
| 4 | 404 | 0.00014498 | 0.00017184 | 685.06 |
| 5 | 505 | 0.00015430 | 0.00018448 | 724.11 |
| **6** | **606** | **0.00014120** | **0.00016914** | 626.64 |
| 7 | 707 | 0.00014645 | 0.00017228 | 668.58 |
| 8 | 808 | 0.00014402 | 0.00016999 | 743.62 |
| 9 | 909 | 0.00014498 | 0.00017349 | 693.10 |
| 10 | 1010 | 0.00014413 | 0.00017069 | 586.70 |
| **Avg** | | **0.00014611** | **0.00017394** | **671.14** |
| **StdDev** | | **0.00000342** | — | — |

**Best expression (Run 6, seed 606):**
```
(((x1 + ((x2 + x4) + x3)) / 4.0562) - ((x3 * x4) - (x5 * x7)))
```

---

## A1 vs A2 Comparison

| Metric | A1 — Canonical GP | A2 — SBGP | Δ |
|---|---|---|---|
| Avg Train MSE | 0.00014812 | **0.00014611** | −1.4% |
| StdDev Train MSE | 0.00000658 | **0.00000342** | **−48%** |
| Best Train MSE | 0.00014257 | **0.00014120** | −1.0% |
| Avg Test MSE | 0.00017889 | **0.00017394** | −2.8% |
| Avg Runtime | 298 s | 671 s | +2.25× |

Structural fitness sharing reduces fitness variance by 48%, confirming improved population diversity at the cost of a 2.25× runtime overhead.

---

## Notes

- Run from the `Assignment 2/` directory so the relative dataset path resolves correctly.
- Seeds are hard-coded in `Main.java` (101, 202, ..., 1010) for full reproducibility.
- Per-generation stats (avg tree size, variety, structural similarity) print to stdout; capture with: `java -jar GP_Assignment2.jar > run_output.txt`
