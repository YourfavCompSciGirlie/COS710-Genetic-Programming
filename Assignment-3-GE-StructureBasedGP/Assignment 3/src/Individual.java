/**
 * A Grammatical Evolution individual: an integer codon genome together with
 * its derived phenotype, derivation signature, and cached fitness values.
 *
 * <p>The genome is a fixed-length {@code int[]} of codons; each codon is a
 * non-negative integer (after mapping &amp; 0x7fffffff) used during BNF
 * derivation as {@code codon mod numProductions} to select the next
 * production rule. Phenotype, signature and validity flag are produced by
 * {@link Grammar#map(int[])}.
 *
 * <p>This class is intentionally a thin data container — all evolutionary
 * operations (selection, crossover, mutation, evaluation) are implemented
 * in {@link GrammaticalEvolution}.
 *
 * <p>COS 710 Assignment 3 – Structure-Based Grammatical Evolution.
 */
public final class Individual {

    /** Integer codon genome. */
    public final int[] genome;

    /** Derived phenotype expression (or placeholder if mapping failed). */
    public final Expression phenotype;

    /** Derivation signature (sequence of production choices). */
    public final int[] signature;

    /** True if the genome mapped to a complete sentence within MAX_WRAPS. */
    public final boolean valid;

    /** Number of codons consumed during mapping. */
    public final int codonsUsed;

    /**
     * Constructs an individual from an int[] genome by performing the
     * derivation through the {@link Grammar} once at construction time.
     */
    public Individual(int[] genome) {
        this.genome = genome;
        Grammar.Derivation d = Grammar.map(genome);
        this.phenotype  = d.phenotype;
        this.signature  = d.signature;
        this.valid      = d.valid;
        this.codonsUsed = d.codonsUsed;
    }
}
