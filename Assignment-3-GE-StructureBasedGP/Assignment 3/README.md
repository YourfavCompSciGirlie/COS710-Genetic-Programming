# COS 710 Assignment 3 — Structure-Based Grammatical Evolution

**Author:** Yohali Malaika Kamangu (u23618583)  
**Language:** Java (JDK 17, no external libraries)

## Project layout

```
Assignment 3/
├── src/
│   ├── Grammar.java               BNF grammar + genotype-to-phenotype mapper
│   ├── Expression.java            Phenotype AST + protected evaluator
│   ├── Individual.java            Codon genome wrapper
│   ├── DatasetLoader.java         CSV loader + sliding window + temporal split
│   ├── GrammaticalEvolution.java  SBGE engine (selection / operators / sharing)
│   └── Main.java                  Entry point — 10 seeds, A1/A2 comparison
├── documentation/
│   └── Residential_Energy_Dataset_UK- 2014-2020.csv
├── out/                           Compiled class files
├── manifest.txt                   Jar manifest (Main-Class: Main)
├── GE_Assignment3.jar             Runnable jar (compiled code)
├── report.tex                     LaTeX source for the report
├── results_output.txt             Captured terminal output of the full run
└── README.md                      This file
```

## Compile and run from source

From the `Assignment 3` directory:

```bash
javac -d out src/*.java
java -cp out Main
```

## Run the pre-built jar

```bash
java -jar GE_Assignment3.jar
```

The CSV must be at `documentation/Residential_Energy_Dataset_UK- 2014-2020.csv`
relative to the working directory.

## Build the jar

```bash
javac -d out src/*.java
jar cfm GE_Assignment3.jar manifest.txt -C out .
```

## Configuration

All parameters are defined as constants at the top of [src/Main.java](src/Main.java):
population size 1000, 100 generations, tournament size 2, crossover rate 0.70,
mutation rate 0.30 (exact-rate operator split, sums to 1.0), per-codon mutation
probability 0.10, genome length 64, max derivation depth 6, σ=0.60, α=0.01.
Ten seeds are used: 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010.

## Summary of results

| Method | Avg Train MSE | StdDev | Best Train MSE | Avg Test MSE | Avg Runtime |
|---|---|---|---|---|---|
| A1 (Canonical GP) | 0.00014812 | 0.00000658 | 0.00014257 | 0.00017889 | 298 s |
| A2 (SBGP)         | 0.00014611 | 0.00000342 | 0.00014120 | 0.00017394 | 671 s |
| A3 (SBGE)         | 0.00019314 | 0.00002406 | 0.00016529 | 0.00023663 |  35 s |

Best evolved expression (A3, Run 4, seed 404):
`(((2.92 * x7) / 4.48) + (x4 / ((1.75 * ((x7 + 0.10) * x7)) * ((8.08 / x3) + 8.86))))`
