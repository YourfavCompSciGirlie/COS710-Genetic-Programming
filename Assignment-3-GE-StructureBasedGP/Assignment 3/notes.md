Page
Page number
2
of 12
Variations of
Genetic
Programming
19 October 2020
Introduction
19 October 2020
• Variations have evolved since the inception of
genetic programming
• These include:
• Linear genetic programming (LGP)
• Cartesian genetic programming (CGP)
• Grammar-based genetic programming (GBGP)
• Grammatical evolution
Linear Genetic Programming (LGP)
19 October 2020
• Programs read and write from registers
• Registers are represented as an array
• Types of registers
• Input registers
• Calculation registers
• Constant registers
• Programs are evolved in C or machine code
Linear Genetic Programming (LGP)
19 October 2020
• Each program has variable length
• Mean square error is often used as a fitness
measure
• Tournament selection is used to choose parents
• Genetic operators must present syntactically and
semantically correct programs
Linear Genetic Programming (LGP)
19 October 2020
Example
Cartesian Genetic Programming (CGP)
19 October 2020
• Each program is a grid of nodes in the form of a
acyclic graph
• Types of nodes
• Function genes
• Connection genes
• Output genes
Cartesian Genetic Programming (CGP)
19 October 2020
Example
TIC PROGRAMMING (CGP) 5
OA = x0 + x1 (1)
OB = x0 ⇤ x1 (2)
OC = x0  x1
x2
0  x1
(3)
OD = x2
0 (4)
are x0 and x1. The function gene lookup table is listed in Table 1


Page
Page number
16
of 16
2.4 Alternative Search Engines 19
yi(t + 1) = xi(t) if, f (xi(t)) > f (yi(t)) (2.5)
After the location of all particles have been updated, a check is made to
determine whether gbest needs to be updated (equation 2.6).
ˆy ∈ (y0, y1, ..., yn)|f (ˆy) = max (f (y0), f (y1), ..., f (yn)) (2.6)
In Grammatical Swarm (GS) the update equations for the swarm algorithm
are as described earlier, with additional constraints placed on the velocity
and particle location dimension values, such that maximum velocities vmax
are bound to ±255, and each dimension is bound to the range [0,255] (de-
noted as cmin and cmax respectively). Note that this is a continuous swarm
algorithm with real-valued particle vectors. The standard GE mapping func-
tion is adopted, with the real-values in the particle vectors being rounded
up or down to the nearest integer value for the mapping process. In the cur-
rent implementation of GS, fixed-length vectors are adopted, within which
it is possible for a variable number of dimensions to be used during the pro-
gram construction genotype-phenotype mapping process. A vector’s elements
(values) may be used more than once if wrapping occurs, and it is also pos-
sible that not all dimensions will be used during the mapping process if a
complete program comprised only of terminal symbols, is generated before
reaching the end of the vector. In this latter case, the extra dimension val-
ues are simply ignored and considered introns that may be switched on in
subsequent iterations.
GS Experimental Findings
A diverse selection of benchmark programs from the literature were tackled
to demonstrate proof of concept for the GS method. The problems included
Santa Fe Ant Trail, a Symbolic Regression instance (x + x2 + x3 + x4), the
3-Multiplexer boolean problem, and Mastermind. The parameters adopted
across the experiments were c1 = c2 = 1.0, wmax = 0.9, wmin = 0.4, cmin
= 0 (minimum value a coordinate may take), cmax = 255 (maximum value a
coordinate may take). In addition, a swarm size of 30 running for 1000 itera-
tions using 100 dimensions is used. The same problems are also tackled with
GE in order to determine how well GS is performing at program generation
in relation to the more traditional variable-length Genetic Algorithm search
engine of standard GE. In an attempt to achieve a relatively fair comparison
of results given the differences between the search engines of Grammatical
Swarm and Grammatical Evolution, we have restricted each algorithm in the
number of individuals they process. Grammatical Swarm running for 1000
iterations with a swarm size of 30 processes 30,000 individuals, therefore,
a standard population size of 500 running for 60 generations is adopted
for Grammatical Evolution. The remaining parameters for Grammatical
20 2 Grammatical Evolution
Evolution are roulette selection, steady state replacement, one-point crossover
with probability of 0.9, and a bit mutation with probability of 0.01.
Table 2.1 provides a summary and comparison of the performance of GS
and GE on each of the problem domains tackled. 100 independent runs were
performed for the data reported. In two out of the four problems GE outper-
forms GS, and GS outperforms GE on the other two problem instances. The
key finding is that the results demonstrate proof of concept that GS can suc-
cessfully generate solutions to problems of interest. In this initial study, we
have not attempted parameter optimisation for either algorithm, but results
and observations of the particle swarm engine suggests that swarm diversity
is open to improvement. We note that a number of strategies have been sug-
gested in the swarm literature to improve diversity [202], and we suspect that
a significant improvement in GS performance can be obtained with the adop-
tion of these measures. Given the relative simplicity of the Swarm algorithm,
the small population sizes involved, and the complete absence of a crossover
operator synonymous with program evolution in GP, it is impressive that
solutions to each of the benchmark problems have been obtained.
Table 2.1 A comparison of the results obtained for Grammatical Swarm and
Grammatical Evolution across all the problems analysed
Mean Best Mean Average Successful
Fitness (Std.Dev.) Fitness (Std.Dev.) Runs
Santa Fe ant
GS 75.24 (16.64) 33.43 (3.69) 43
GE 80.18 (13.79) 46.43 (11.18) 58
Multiplexer
GS 0.97 (0.05) 0.87 (0.01) 79
GE 0.95 (0.06) 0.88 (0.04) 56
Symbolic Regression
GS 0.31 (0.35) 0.07 (0.02) 20
GE 0.88 (0.30) 0.28 (0.28) 85
Mastermind
GS 0.91 (0.04) 0.88 (0.01) 18
GE 0.90 (0.03) 0.89 (0.00) 10
2.4.2 Grammatical Differential Evolution
Differential evolution (DE) [209, 210, 211, 175] is a population-based search
algorithm. The algorithm draws inspiration from the field of Evolutionary
Computation, as it embeds implicit concepts of mutation, recombination and
fitness-based selection, to evolve from an initial randomly generated popu-
lation to a solution to a problem of interest. It also borrows principles from
2.4 Alternative Search Engines 21
Social Algorithms through the manner in which new individuals are gener-
ated. Unlike the binary chromosomes typical of GAs, an individual in DE is
generally comprised of a real-valued chromosome.
Although several DE algorithms exist we only describe one version of the
algorithm based on the DE/rand/1/bin scheme [209]. The different variants
of the DE algorithm are described using the shorthand DE/x/y/z, where x
specifies how the base vector to be perturbed is chosen (rand if it is randomly
selected or best if the best individual is selected), y is the number of difference
vectors used, and z denotes the crossover scheme used (bin for crossover based
on independent bi-nominal experiments, and exp for exponential crossover).
At the start of this algorithm, a population of N , d-dimensional vectors
Xj = (xi1 , xi2, . . . , xid), j = 1, . . . , n, is randomly initialised and evaluated
using a fitness function f . During the search process, each individual (j) is
iteratively refined. The modification process has three steps:
i. Create a variant solution, using randomly selected members of the pop-
ulation.
ii. Create a trial solution, by combining the variant solution with j (crossover
step).
iii. Perform a selection process to determine whether the trial solution re-
places j in the population.
Under the mutation operator, for each vector Xj (t), a variant solution Vj (t+1)
is obtained using equation 2.7:
Vj (t + 1) = Xm(t) + F (Xk(t) − Xl(t)) (2.7)
where k, l, m ∈ 1, . . . , N are mutually different, randomly selected indices,
and all the indices  = j (Xm is referred to as the base vector, and Xk(t)−Xl(t)
is referred to as a difference vector). Variants on this step include the use of
more than three individuals from the population, and/or the inclusion of the
highest-fitness point in the population as one of these individuals [209]. The
difference between vectors Xk and Xl is multiplied by a scaling parameter
F (typically, F ∈ (0, 2]). The scaling factor controls the amplification of
the difference between Xk and Xl, and is used to avoid stagnation of the
search process. Following the creation of the variant solution, a trial solution
Uj (t + 1) = (uj1, uj2, . . . , ujd) is obtained from equation 2.8.
Ujn(t + 1) =
{
Vjn, if (rand ≤ CR) or (j = rnbr(i)) ;
Xjn, if (rand > CR) and (j  = rnbr(i)). (2.8)
where n = 1, 2, . . . , d, rand is drawn from a uniform random number gener-
ator in the range (0,1), CR is the user-specified crossover constant from the
range (0,1), and rnbr(i) is a randomly chosen index chosen from the range
(1, 2, . . . , n). The random index is used to ensure that the trial solution differs
by at least one component from Xi(t). The resulting trial solution replaces its
22 2 Grammatical Evolution
Fig. 2.5 A represen-
tation of the Differen-
tial Evolution variety-
generation process. The
value of F is set at 0.50.
In a simple 2-d case,
the child of particle Xj
can end up in any of
three positions. It may
end up at either of the
two positions X∗
j , or at
the position of particle
Vj (t + 1).
X j*
X j
X k
X m
V j (t+1)
(F=0.5)
X j*
X l
predecessor , if it has higher fitness (a form of selection), otherwise the prede-
cessor survives unchanged into the next iteration of the algorithm (equation
2.9).
Xi(t + 1) =
{
Ui(t + 1), if f (Ui(t + 1)) < f (Xi(t));
Xi(t), otherwise. (2.9)
Fig. 2.5 provides a graphic of the adaptive process of GDE. The DE algorithm
has three parameters, the population size (N), the crossover rate (CR), and
the scaling factor (F). Higher values of CR tend to produce faster convergence
of the population of solutions. Typical values for these parameters are in the
range, N=50-100 (or ten times the number of dimensions in a solution vector),
CR=0.8-0.9 and F=0.3-0.5.
Grammatical Differential Evolution (GDE) adopts a Differential Evolu-
tion learning algorithm coupled to a Grammatical Evolution (GE) genotype-
phenotype mapping to generate programs in an arbitrary language. The stan-
dard GE mapping function is adopted with the real-values in the vectors being
rounded up or down to the nearest integer value, for the mapping process. In
the current implementation of GDE, fixed-length vectors are adopted within
which it is possible for a variable number of elements to be required during
the program construction genotype-phenotype mapping process. A vector’s
values may be used more than once if the wrapping operator is used, and in
the opposite case it is possible that not all elements will be used during the
mapping process if a complete program comprised only of terminal symbols
is generated before reaching the end of the vector. In this latter case, the
extra element values are simply ignored and considered introns that may be
switched on in subsequent iterations.
GDE Experimental Findings
The same diverse set of problems are tackled with GDE as with GS, including
an instance of Symbolic Regression (x+x2 +x3 +x4), the Santa Fe Ant Trail,
2.5 Applications of GE 23
boolean 3-Multiplexer, and Mastermind. The parameters adopted across the
following experiments are Params of GDE....popsize 500, 100 iterations, strlen
100, F=0.9, CR=1.0, DE/best/1/exp. Gene values are bound to the range
[0 → 255].
The same problems are also tackled with Grammatical Evolution in order
to get some indication of how well GDE is performing at program generation
in relation to the more traditional variable-length Genetic Algorithm-driven
search engine of standard GE. a standard population size of 500 running
for 60 generations is adopted for Grammatical Evolution. The remaining
parameters for Grammatical Evolution are roulette selection, steady state
replacement, one-point crossover with probability of 0.9, and a bit mutation
with probability of 0.01.
Table 2.2 provides a summary and comparison of the performance of Gram-
matical Differential Evolution, and Grammatical Evolution on each of the
problem domains tackled. The reported results are averaged over 50 indepen-
dent runs. In three out of the four problems Grammatical Evolution outper-
forms GDE. The key finding is that the results demonstrate proof of concept
that GDE can successfully generate solutions to problems of interest.
Table 2.2 A comparison of the results obtained for Grammatical Differential Evo-
lution and Grammatical Evolution across all the problems analysed
Santa Fe Symbolic
Ant Multiplexer Regression Mastermind
GDE/rand/1/bin 10 23 6 0
GDE/best/1/exp 7 27 4 0
GDE/rand-to-best/1/exp 9 27 4 0
GDE/rand-to-best/1/bin 7 25 5 0
GE 17 15 24 3
2.5 Applications of GE
Since its inception GE has received considerable attention and been applied
to a wide variety of problem domains. Early studies saw GE being successfully
applied to symbolic-regression problems [190], the evolution of trigonometric
identities [192], the evolution of caching algorithms [145], and behavioural
robotics [146, 147].
GE was extended and applied to the domain of surface design in [87] where
it was combined with GENR8, a surface design tool that uses an evolution-
ary process to adapt surfaces to meet a user’s specification. In the field of
Bioinformatics, GE was applied to recognise eukaryotic promotors [161] that
help in the identification of biological genes. While in [137] GE was used to
evolve neural networks for feature selection in genetic epidemiology.
In the area of sound synthesis and analysis Ortega et al. [168] use GE
to automatically generate compositions which were comparable to human
24 2 Grammatical Evolution
composed works. In [109] GE was tasked with evolving phonological rules
which can be used to translate text into graphemes and recognition of sounds
as words.
Fractal curves of a high dimensionality were evolved in [169] and in [130]
Petri-Net models of complex systems were evolved. GE has also been used in
the design Logos [164].
In the financial domain, GE has been applied in a number of areas [24], with
studies in foreign-exchange trading [17, 18, 22], bankruptcy and corporate
analysis [16, 19, 20], and credit classification [21, 23]. Specific to the trading
of market indices, which is examined in this book, studies have also been
conducted [149, 150, 15] where the problem is examined in a static manner.
2.6 Conclusion
This chapter introduced Grammatical Evolution and described its mapping
process and how the genetic operators of mutation and crossover are im-
plemented along with their effects. Some noteable applications of GE were
outlined.
In addition to the various papers on the subject (see the GP Bibliog-
raphy [115]), further information on GE can be found from http://www.
grammatical-evolution.org including links to various software implemen-
tations of GE. A version of GE in Java, GEVA [76], has recently been released
by our Natural Computing Research & Applications group at University Col-
lege Dublin. This is available directly from http://ncra.ucd.ie/geva/.
Subsequent chapters will explore the features of GE and its potential for
use in dynamic environments. Before we begin a more in-depth analysis of
GE in dynamic environments, in the following chapter we first provide an
overview of the research to date on Evolutionary Computation in these non-
stationary domains.

