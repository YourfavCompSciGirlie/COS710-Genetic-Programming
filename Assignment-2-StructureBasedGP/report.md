# COS 710 Assignment 2: Structure-Based Genetic Programming for Regression

**Student:** [Student Name]  
**Student Number:** u23618583  
**Date:** April 2026  
**Dataset:** Residential Energy Usage UK (2014–2020)

---

## 1. Description of the Dataset

The dataset originates from the UK residential energy sector and covers a
nearly five-year period from 31 December 2014 to 2020. It contains 201,604
records collected at 15-minute intervals during that period. The following
attributes are recorded for each timestamp:

| Column | Description | Unit |
|---|---|---|
| `utc_timestamp` | Date and time of the record in UTC | — |
| `Electricity_load` | Residential electricity demand | kW |
| `Residential_electricity_price` | Residential electricity price | GBP |
| `Residential_solar_generation` | Solar power generation | kW |
| `Residential_wind_generation` | Wind power generation | kW |
| `Temperature` | Ambient temperature | °C |
| `Relative Humidity` | Relative humidity | % |

Only the `Electricity_load` column is used for this regression task. The
problem is formulated as a univariate time-series prediction: given the
electricity load values at the *n* most recent 15-minute intervals, predict
the load at the next interval. A sliding window of size 7 (105 minutes of
history) is applied to build the supervised learning dataset:

- Input variables: x₁ = Load(t−1), x₂ = Load(t−2), …, x₇ = Load(t−7)
- Target: Load(t)

To keep each run to a manageable duration, the first 100,000 valid rows are
loaded from the CSV file, yielding 99,993 windowed examples. An 80/20
temporal split (preserving chronological order) produces 79,994 training
examples and 19,999 test examples. No temporal shuffling is applied, as
shuffling would introduce data leakage in a time-series setting.

Electricity load values lie predominantly in the range [0.04, 0.15] kW with
occasional higher peak readings. The series exhibits clear circadian and
weekly seasonality typical of residential consumption patterns.

---

## 2. Description of the Performance Metric

**Mean Squared Error (MSE)** is the primary fitness metric:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (\hat{y}_i - y_i)^2$$

where $\hat{y}_i$ is the tree's prediction for example $i$ and $y_i$ is the
true load value. MSE is used for all three purposes — training fitness,
test evaluation, and summary statistics — because:

1. It is differentiable and penalises large errors superlinearly, strongly
   discouraging outlier predictions.
2. Its units (kW²) are consistent with the load scale, making cross-run
   comparisons straightforward.
3. Lower is better, matching the minimisation structure of GP's fitness
   function.

**Root Mean Squared Error (RMSE)** = $\sqrt{\text{MSE}}$ is noted in
discussion to interpret results in the same physical units as the load (kW).

**Parsimony pressure** is incorporated into the training fitness to
discourage bloated trees:

$$\text{rawFitness} = \text{MSE} \times (1 + \lambda \cdot |T|)$$

where $|T|$ is the tree's node count and $\lambda = 0.001$.

**Hits criterion:** A prediction is a "hit" if $|\hat{y}_i - y_i| < 0.01$ kW
(approximately 10% of a typical reading). The hit count for the best
individual is recorded per generation.

**Success predicate:** A run is considered successful if the best training MSE
found falls below 0.00020.

---

## 3. How Structure is Incorporated into Canonical GP

### 3.1 Motivation

Canonical GP (Assignment 1) drives the search solely through behavioural
fitness — how well a tree's outputs match the training targets. This
behaviour-only selection pressure tends to produce structurally homogeneous
populations: once a particular pattern of operators and terminals proves
beneficial, it proliferates rapidly, reducing the diversity needed to
continue effective exploration. Assignment 2 addresses this by augmenting
the fitness function with a structural diversity component.

### 3.2 The Similarity Index

Following the lecture slide example, structural similarity is measured by
**counting the number of nodes that are the same from the root** — a
top-down prefix node matching scheme:

```
prefixMatchCount(T₁, T₂):
  if root₁.label ≠ root₂.label → return 0
  count = 1
  for each child pair (c₁, c₂):
      count += prefixMatchCount(c₁, c₂)
  return count
```

The count is normalised by the size of the larger tree to yield a
dimensionless similarity score in [0, 1]:

$$\text{sim}(T_1, T_2) = \frac{\text{prefixMatchCount}(\text{root}_1, \text{root}_2)}{\max(|T_1|, |T_2|)}$$

- Two identical trees return 1.0.
- Two trees with different root labels return 0.0.
- Trees `(+ x1 x2)` and `(+ x1 x3)` return 2/3 ≈ 0.667 (root `+` and left child `x1` match; right children differ).

### 3.3 Structural Fitness Sharing

Structure is incorporated **as part of the fitness function** — the first of
the four options described in the SBGP lecture slides. After computing the
standard MSE-based raw fitness each generation, each individual's fitness is
adjusted upward (worsened) in proportion to how many structurally similar
peers it has in the population:

$$\text{adjustedFitness}[i] = \text{rawFitness}[i] \times (1 + \alpha \cdot \text{nicheCount}[i])$$

where:

- $\text{nicheCount}[i]$ = number of sampled peers $j$ with $\text{sim}(T_i, T_j) \geq \sigma$
- $\sigma = 0.60$ is the **similarity threshold** (a tree must share at least
  60% of its top-down node structure with individual $i$ to counted as a
  structural neighbour)
- $\alpha = 0.01$ is the **sharing coefficient** (scales the penalty so the
  structural component complements rather than dominates the behavioural
  signal)

To avoid the O(n²) cost of all-pairs comparison (1,000 × 999 ≈ 10⁶
comparisons per generation), each individual's niche count is estimated by
sampling 50 random peers, giving 50,000 structural comparisons per
generation at a fraction of the full cost.

### 3.4 How This Promotes Diversity and Directs the Search

Individuals that are structurally crowded (many similar neighbours) receive a
fitness penalty, making them less likely to be selected as parents or
preserved by elitism. This discourages the population from collapsing into a
structurally uniform cluster and provides sustained selection pressure for
exploring structurally distinct regions of the programme space — regions that
would be abandoned in canonical GP. The mechanism is conceptually analogous
to sharing-based niching in genetic algorithms but operates on tree structure
rather than genotype distance.

---

## 4. Representation, Terminal Set, and Function Set

### 4.1 Representation

Each GP individual is a **full expression tree** rooted at a binary function
node. Nodes store a label (operator or terminal), arity, an ordered children
list, and a parent pointer. The parent pointer enables O(depth) depth queries
during mutation and crossover, which is critical for the depth-bounded bloat
control mechanism.

### 4.2 Terminal Set

| Terminal | Meaning |
|---|---|
| x₁ | Load(t−1) — most recent lag |
| x₂ | Load(t−2) |
| x₃ | Load(t−3) |
| x₄ | Load(t−4) |
| x₅ | Load(t−5) |
| x₆ | Load(t−6) |
| x₇ | Load(t−7) — oldest lag |
| ERC | Ephemeral Random Constant: $U(-5.0, 5.0)$, 30% probability at terminal creation |

### 4.3 Function Set

| Operator | Arity | Note |
|---|---|---|
| `+` | 2 | Addition |
| `-` | 2 | Subtraction |
| `*` | 2 | Multiplication |
| `/` | 2 | Protected division: $a\,/\,(|b|+0.0001)$ |

Protected division ensures the function set is closed over the reals,
preventing infinite or NaN outputs from divide-by-zero.

---

## 5. Initial Population Generation

The initial population is generated using the **Grow** method:

- The root of every tree is forced to be a function node (no trivial
  single-terminal individuals at startup).
- At each non-root node, a function node is chosen with 50% probability and a
  terminal node with 50%, unless the maximum depth (6) is reached, in which
  case a terminal is forced.
- Up to three generation attempts are made per population slot; if a
  duplicate infix expression is produced, the attempt is discarded and a new
  tree is grown. This promotes initial diversity without enforcing a strict
  uniqueness constraint (the third attempt is always accepted even if
  duplicate, to avoid an infinite loop on degenerate fitness landscapes).

---

## 6. Fitness Function and Fitness Evaluation

### 6.1 Raw Fitness (Behavioural)

For each training example $i$ with input vector $\mathbf{x}_i$ and true load
$y_i$, the tree $T$ is evaluated to produce $\hat{y}_i = T(\mathbf{x}_i)$.
If any prediction is `NaN` or `±Infinity` (possible for extreme ERC values in
deeply nested divisions), the individual is assigned a fitness of
`Double.MAX_VALUE` and is effectively removed from the gene pool.

Otherwise, raw fitness combines MSE with parsimony pressure:

$$\text{rawFitness}(T) = \underbrace{\frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2}_{\text{MSE}} \times (1 + 0.001 \cdot |T|)$$

### 6.2 Adjusted Fitness (Behavioural + Structural)

Following raw fitness evaluation, **structural fitness sharing** is applied
(see Section 3) to produce the adjusted fitness array used in selection and
elitism. The adjusted fitness penalises structurally crowded individuals while
leaving individuals in uncrowded structural niches essentially unchanged.

Raw fitness is retained as the basis for tracking and reporting the best
individual, ensuring reported MSE values reflect actual prediction quality
without structural inflation.

---

## 7. Selection Method

**Tournament selection** with tournament size $k = 2$ is used to select
parents for crossover and mutation. Two candidates are drawn uniformly at
random (with replacement) from the population; the one with the lower
*adjusted* fitness wins and is used as a parent.

A small tournament size reduces selection pressure compared to higher values
of $k$, helping avoid premature convergence. Since adjusted fitness already
signals structural crowding, the lower selection pressure also allows
structurally diverse (but behaviourally competitive) individuals more
opportunity to contribute offspring.

---

## 8. Genetic Operators

### 8.1 Subtree Crossover

**Rate:** 90% of non-elite offspring slots are produced by crossover (exact
count: round(980 × 0.90) = 882 offspring per generation).

A random structural point in a copy of parent 1's tree is selected; the
subtree rooted at that point is replaced with a deep copy of a random subtree
from parent 2. One offspring is produced per crossover operation. Bloat
control (trimToDepth, max depth 6) is applied immediately after the swap.

### 8.2 Subtree Mutation

**Rate:** 10% of non-elite offspring slots (98 per generation).

A random node in a copy of the parent tree is selected as the mutation point.
The depth remaining from that node to the maximum depth is computed; a freshly
grown subtree of at most that depth is generated and grafted in place of the
old subtree. Depth limit enforcement is applied as a safety net.

### 8.3 Exact Operator Rates via Fisher-Yates Shuffle

To guarantee that exactly 882 crossover offspring and 98 mutation offspring
are produced per generation (not approximately, as would occur with
per-individual probability rolls), a slot-type array is constructed:
the first 882 entries are set to 0 (crossover) and the remaining 98 to 1
(mutation). The array is then shuffled in-place using Fisher-Yates, randomising
the order in which operators are applied across the 980 non-elite slots.

### 8.4 Elitism

The top 2% of individuals by *adjusted* fitness (20 individuals) are copied
unchanged into the next generation, ensuring the best-quality diverse
solutions found so far are never lost.

---

## 9. Experimental Setup

### 9.1 Parameter Values

| Parameter | Value |
|---|---|
| Population size | 1,000 |
| Max tree depth | 6 |
| Generations | 100 |
| Tournament size | 2 |
| Crossover rate | 0.90 |
| Mutation rate | 0.10 |
| Elitism rate | 2% (20 individuals) |
| Window size | 7 |
| Training ratio | 0.80 (80/20 split) |
| Parsimony coefficient (λ) | 0.001 |
| ERC range | [−5.0, 5.0] |
| ERC probability | 30% |
| **Similarity threshold (σ)** | **0.60** |
| **Sharing coefficient (α)** | **0.01** |
| **Structural sample size** | **50** |
| Data sample size | 100,000 rows |
| Training examples | 79,994 |
| Test examples | 19,999 |
| Independent runs | 10 |
| Random seeds | 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010 |

### 9.2 Technical Specifications

| Specification | Detail |
|---|---|
| Language | Java JDK 17 |
| External libraries | None (standard library only) |
| Machine | Apple MacBook (macOS) |
| Architecture | ARM64 / Apple Silicon |
| JVM | OpenJDK 17 |

---

## 10. Results

All 10 runs were completed successfully. Results are reported using the raw
training MSE (without the parsimony component) as the primary metric. The
success predicate (train MSE < 0.00020) was met by all runs.

### 10.1 Per-Run Results

| Run | Seed | Train MSE | Test MSE | Time (s) |
|---|---|---|---|---|
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
| **Avg** | | **0.00014611** | **0.00017394** | 671.14 |
| **StdDev** | | **0.00000342** | — | — |
| **Best** | | **0.00014120** | **0.00016914** | — |

### 10.2 Per-Generation Statistics (Best Run)

| Gen | Avg Train MSE | Avg TreeSize | Hits | Variety | Avg Struct Sim |
|---|---|---|---|---|---|
| 10 | 0.02190210 | 4.74 | 43063 | 417 | 0.1042 |
| 20 | 0.02228552 | 9.16 | 44293 | 694 | 0.0384 |
| 30 | 0.02688157 | 13.85 | 44293 | 836 | 0.0380 |
| 40 | 0.02068111 | 17.19 | 44416 | 917 | 0.0697 |
| 50 | 0.02321578 | 21.37 | 44293 | 909 | 0.0579 |
| 60 | 0.03647585 | 26.12 | 44551 | 954 | 0.1328 |
| 70 | 0.02534679 | 26.63 | 44490 | 953 | 0.1670 |
| 80 | 0.02623639 | 26.26 | 44579 | 960 | 0.1867 |
| 90 | 0.03123360 | 26.29 | 44579 | 962 | 0.2091 |
| 100 | 0.02209658 | 26.54 | 44579 | 951 | 0.2195 |

### 10.3 Best Evolved Expression

The best expression was found in Run 6 (seed 606, train MSE = 0.00014120):

```
(((x1 + ((x2 + x4) + x3)) / 4.0562) - ((x3 * x4) - (x5 * x7)))
```

This expression is interpretable: the first term computes a weighted average of the
four most recent lag values (dividing by ≈4 approximates a 4-step moving average).
The second term adds a small correction based on pairwise products of older lags,
capturing non-linear interactions between distant time steps.

### 10.4 Discussion

The SBGP implementation applies structural fitness sharing throughout every
generation. The `Avg Struct Sim` column in the per-generation table provides
direct evidence of how structural diversity evolves over the run. A lower
average structural similarity indicates that the population maintains diverse
tree shapes, while a rising value would suggest structural convergence.

Unlike canonical GP where diversity is maintained only via the Variety column
(unique expression count), SBGP actively penalises structurally crowded
individuals during selection, preserving diversity more deliberately.

The parsimony coefficient (λ = 0.001) discourages tree bloat. The fitness
sharing coefficient (α = 0.01) is intentionally small so the structural
penalty complements rather than overwhelms the MSE signal.

---

## 11. Runtimes

The average runtime across the 10 SBGP runs is 671 seconds (11.2 minutes) per run,
compared to Assignment 1's average of 298 seconds — approximately 2.25× slower.
Per-run times ranged from 541 s (Run 2) to 780 s (Run 3). The overhead comes from
the structural fitness sharing step: 50,000 prefix-match traversals per generation ×
100 generations = 5,000,000 additional structural comparisons per run.

---

## 12. Comparison with Assignment 1

| | Avg Train MSE | StdDev | Best Train MSE | Avg Test MSE |
|---|---|---|---|---|
| **A1 — Canonical GP** | 0.00014812 | 0.00000658 | 0.00014257 | 0.00017889 |
| **A2 — Structure-Based GP** | **0.00014611** | **0.00000342** | **0.00014120** | **0.00017394** |

### Discussion

SBGP outperforms canonical GP on every reported metric:

- **Average train MSE** improved by 1.4% (0.00014812 → 0.00014611)
- **Standard deviation** fell by 48% (0.00000658 → 0.00000342) — the most
  striking result; SBGP is far more consistent across seeds
- **Best train MSE** improved by 0.96% (0.00014257 → 0.00014120)
- **Average test MSE** improved by 2.8% (0.00017889 → 0.00017394)
- **Test/train ratio** improved from ≈1.208× (A1) to ≈1.190× (A2)
- **Success rate:** 10/10 for both systems

The reduction in variance is particularly notable. By penalising structurally
crowded individuals, SBGP maintains structural diversity (Avg Struct Sim < 0.25
throughout all generations), enabling the search to explore a broader variety of
expression templates. This prevents the premature structural convergence that
causes high run-to-run variance in canonical GP.

The trade-off is runtime: SBGP requires approximately 2.25× more computation
per run (avg 671 s vs avg 298 s for A1). For problems where consistency and
marginal accuracy gains are valued over speed, SBGP offers a worthwhile improvement.
