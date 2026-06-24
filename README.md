# COS710 – Genetic Programming

> **Official Module:** Artificial Intelligence (I) 710 (COS 710)  
> **University of Pretoria** – Honours | NQF Level 08 | 15 credits  
> **Lecturer:** Prof. Nelishia Pillay  
> **Semester 1, 2026**

![GitHub repo size](https://img.shields.io/github/repo-size/YourfavCompSciGirlie/COS710-Genetic-Programming)
![GitHub language count](https://img.shields.io/github/languages/count/YourfavCompSciGirlie/COS710-Genetic-Programming)
![GitHub top language](https://img.shields.io/github/languages/top/YourfavCompSciGirlie/COS710-Genetic-Programming)

---

## Module Overview

This module provides an in‑depth study of **Genetic Programming (GP)**, a sub‑field of Evolutionary Computation.  
Topics covered include:

- The standard GP algorithm (initialisation, fitness, selection, genetic operators)
- Symbolic regression & data classification
- Grammar‑based GP & **Grammatical Evolution (GE)**
- Advanced topics: iteration, recursion, memory, data structures, modularisation
- Recent developments: **Structure‑Based GP (SBGP)** & transfer learning

All assignments are implemented as individual programming tasks in **Java**, applying the techniques studied in class to predict UK residential electricity load.

---

## Assignments Overview

| # | Assignment | Method | Key Focus | Due Date |
|:-|------------|--------|-----------|----------|
| 1 | [Regression](./Assignment-1-Regression) | Canonical Tree‑Based GP | Symbolic regression with parsimony pressure | 16 March 2026 |
| 2 | [Structure‑Based GP](./Assignment-2-StructureBasedGP) | SBGP | Structural fitness sharing (prefix matching) | 20 April 2026 |
| 3 | [GE + SBGP](./Assignment-3-GE-StructureBasedGP) | SBGE | Grammatical Evolution + structural sharing | 24 May 2026 |

Each assignment folder contains:
- Complete source code (`/src`)
- Pre‑built executable JAR file
- Detailed `README.md` with compilation/execution instructions
- Assignment report (PDF + LaTeX source)
- Dataset (`/documentation`)

---

## Key Results Summary

A comparison of the three approaches on the same regression task (10 independent runs):

| Method | Avg Train MSE | Best Train MSE | Avg Test MSE | Avg Runtime |
|--------|---------------|----------------|--------------|-------------|
| **A1 – Canonical GP** | 0.00014812 | 0.00014257 | 0.00017889 | ~5 min |
| **A2 – SBGP** | 0.00014611 | 0.00014120 | 0.00017394 | ~11 min |
| **A3 – SBGE** | 0.00019314 | 0.00016529 | 0.00023663 | ~35 sec |

> **Insight:** Structure‑based GP (A2) improved test generalisation and reduced variance by **48%** compared to canonical GP, at the cost of runtime. Grammatical Evolution (A3) was significantly faster but traded off some accuracy.

---

## Getting Started

### Prerequisites
- **Java JDK 17+** (Download from [Adoptium](https://adoptium.net/temurin/releases/?version=17))
- No external libraries — all code uses pure standard Java.

### Running an Assignment
Navigate to the desired assignment folder and run its pre‑built JAR:

```bash
cd Assignment-1-Regression
java -jar GP_Assignment1.jar