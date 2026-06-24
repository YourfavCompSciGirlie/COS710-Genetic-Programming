import java.util.ArrayList;
import java.util.List;

/**
 * BNF grammar and genotype-to-phenotype mapper for Grammatical Evolution.
 *
 * <h2>Grammar (BNF)</h2>
 * <pre>
 *   &lt;expr&gt;  ::= &lt;expr&gt; &lt;op&gt; &lt;expr&gt;          [production 0]
 *              | ( &lt;expr&gt; &lt;op&gt; &lt;expr&gt; )        [production 1]
 *              | &lt;var&gt;                        [production 2]
 *              | &lt;const&gt;                      [production 3]
 *   &lt;op&gt;    ::= + | - | * | /
 *   &lt;var&gt;   ::= x1 | x2 | x3 | x4 | x5 | x6 | x7
 *   &lt;const&gt; ::= &lt;digit&gt;.&lt;digit&gt;&lt;digit&gt;          [single production]
 *   &lt;digit&gt; ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
 * </pre>
 *
 * <h2>Mapping</h2>
 * Leftmost depth-first derivation. Each non-terminal consumes one codon;
 * the chosen production is {@code codon mod numProductions}. If the codon
 * stream is exhausted, the read pointer wraps to index 0; up to
 * {@link #MAX_WRAPS} wraps are allowed before the individual is declared
 * invalid (phenotype incomplete, fitness penalised).
 *
 * <p>To guarantee termination a depth-bounded fallback is applied: once
 * the derivation depth reaches {@link #MAX_DEPTH}, {@code <expr>} is
 * restricted to its terminal productions ({@code <var>} or {@code <const>}).
 * The depth cap matches the GP tree-depth limit used in Assignments 1 and 2,
 * ensuring a fair search-space comparison.
 *
 * <h2>Derivation signature</h2>
 * During mapping the sequence of (non-terminal, production) pairs is recorded
 * as a packed int array. This signature represents the <em>structural
 * skeleton</em> of the individual independently of its numerical constants —
 * the basis for structural similarity used by the SBGE fitness-sharing
 * mechanism (see {@link GrammaticalEvolution}).
 *
 * <p>COS 710 Assignment 3 – Structure-Based Grammatical Evolution.
 */
public final class Grammar {

    // ─── Non-terminal IDs ─────────────────────────────────────────────────────
    public static final int NT_EXPR  = 0;
    public static final int NT_OP    = 1;
    public static final int NT_VAR   = 2;
    public static final int NT_DIGIT = 3;

    /** Number of input variables (last seven lag features x1..x7). */
    public static final int NUM_VARS = 7;

    /** Number of productions per non-terminal. */
    public static final int[] NUM_PRODUCTIONS = {
            4,        // <expr>
            4,        // <op>
            NUM_VARS, // <var>
           10         // <digit>
    };

    /** Maximum codon-stream wraps before mapping is declared failed. */
    public static final int MAX_WRAPS = 2;

    /** Maximum derivation depth (matches GP MAX_DEPTH = 6 from A1/A2). */
    public static final int MAX_DEPTH = 6;

    private Grammar() { /* static utility */ }

    /**
     * Container returned by {@link #map(int[])}: phenotype AST, derivation
     * signature for structural similarity, validity flag and codons used.
     */
    public static final class Derivation {
        public final Expression phenotype;
        public final int[]      signature;
        public final boolean    valid;
        public final int        codonsUsed;

        Derivation(Expression phenotype, int[] signature,
                   boolean valid, int codonsUsed) {
            this.phenotype  = phenotype;
            this.signature  = signature;
            this.valid      = valid;
            this.codonsUsed = codonsUsed;
        }
    }

    /** Mutable mapping state shared across recursive calls. */
    private static final class State {
        final int[]         genome;
        final List<Integer> signature = new ArrayList<>(64);
        int                 idx       = 0;
        int                 wraps     = 0;
        boolean             failed    = false;

        State(int[] genome) { this.genome = genome; }

        int nextCodon() {
            if (idx >= genome.length) {
                wraps++;
                if (wraps > MAX_WRAPS) { failed = true; return 0; }
                idx = 0;
            }
            return genome[idx++] & 0x7fffffff;
        }
    }

    // ─── Public entry point ───────────────────────────────────────────────────

    /**
     * Maps an integer codon genome to a phenotype AST via leftmost derivation.
     * If the codon stream cannot be mapped within {@link #MAX_WRAPS} wraps,
     * {@code Derivation.valid} is {@code false}; a placeholder constant is
     * substituted so downstream evaluation is still safe.
     */
    public static Derivation map(int[] genome) {
        if (genome == null || genome.length == 0) {
            return new Derivation(Expression.constant(0.0), new int[0], false, 0);
        }
        State s = new State(genome);
        Expression e = expr(s, 0);
        if (s.failed) {
            return new Derivation(Expression.constant(0.0),
                                  toIntArray(s.signature), false, s.idx);
        }
        return new Derivation(e, toIntArray(s.signature), true, s.idx);
    }

    // ─── Recursive production rules ───────────────────────────────────────────

    private static Expression expr(State s, int depth) {
        int codon = s.nextCodon();
        if (s.failed) return Expression.constant(0.0);

        int prod;
        if (depth >= MAX_DEPTH) {
            // Depth fallback: terminal productions only (var or const).
            prod = 2 + (codon % 2);
        } else {
            prod = codon % NUM_PRODUCTIONS[NT_EXPR];
        }
        s.signature.add(NT_EXPR * 16 + prod);

        switch (prod) {
            case 0:   // <expr> <op> <expr>
            case 1: { // ( <expr> <op> <expr> )  — same AST structurally
                Expression left   = expr(s, depth + 1);
                int        opCode = op(s);
                Expression right  = expr(s, depth + 1);
                return Expression.op(opCode, left, right);
            }
            case 2:
                return var(s);
            case 3:
            default:
                return cons(s);
        }
    }

    private static int op(State s) {
        int codon = s.nextCodon();
        if (s.failed) return Expression.OP_ADD;
        int prod = codon % NUM_PRODUCTIONS[NT_OP];
        s.signature.add(NT_OP * 16 + prod);
        return prod; // 0..3 maps directly to OP_ADD..OP_DIV
    }

    private static Expression var(State s) {
        int codon = s.nextCodon();
        if (s.failed) return Expression.variable(0);
        int prod = codon % NUM_PRODUCTIONS[NT_VAR];
        s.signature.add(NT_VAR * 16 + prod);
        return Expression.variable(prod);
    }

    private static Expression cons(State s) {
        int d1 = digit(s);
        int d2 = digit(s);
        int d3 = digit(s);
        double v = d1 + d2 / 10.0 + d3 / 100.0;
        return Expression.constant(v);
    }

    private static int digit(State s) {
        int codon = s.nextCodon();
        if (s.failed) return 0;
        int prod = codon % NUM_PRODUCTIONS[NT_DIGIT];
        s.signature.add(NT_DIGIT * 16 + prod);
        return prod;
    }

    // ─── Helpers ──────────────────────────────────────────────────────────────

    private static int[] toIntArray(List<Integer> list) {
        int[] arr = new int[list.size()];
        for (int i = 0; i < arr.length; i++) arr[i] = list.get(i);
        return arr;
    }

    /**
     * Number of leading positions where two derivation signatures agree.
     * Used as the structural-similarity primitive in fitness sharing.
     */
    public static int prefixMatchCount(int[] a, int[] b) {
        int n = Math.min(a.length, b.length);
        int i = 0;
        while (i < n && a[i] == b[i]) i++;
        return i;
    }
}
