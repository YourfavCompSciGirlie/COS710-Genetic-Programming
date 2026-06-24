/**
 * Expression AST node for the phenotype of a GE individual.
 *
 * <p>An {@code Expression} is a binary tree of arithmetic operations
 * produced by mapping a GE codon genome through the {@link Grammar}'s BNF
 * rules. Internal nodes hold a binary operator (+, −, ×, ÷); leaf nodes
 * hold either a variable index (0..6 → x1..x7) or an ephemeral constant.
 *
 * <p>Evaluation uses protected division to keep the closure property:
 * a division by anything with absolute value below 1e-9 returns 1.0.
 *
 * <p>COS 710 Assignment 3 – Structure-Based Grammatical Evolution.
 */
public final class Expression {

    // Node types
    public static final int VAR   = 0;
    public static final int CONST = 1;
    public static final int OP    = 2;

    // Operator codes
    public static final int OP_ADD = 0;
    public static final int OP_SUB = 1;
    public static final int OP_MUL = 2;
    public static final int OP_DIV = 3;

    public final int        type;
    public final int        ivalue;      // variable index or operator code
    public final double     dvalue;      // constant value (when type == CONST)
    public final Expression left;
    public final Expression right;

    private Expression(int type, int ivalue, double dvalue,
                       Expression left, Expression right) {
        this.type   = type;
        this.ivalue = ivalue;
        this.dvalue = dvalue;
        this.left   = left;
        this.right  = right;
    }

    // ─── Factory methods ──────────────────────────────────────────────────────

    public static Expression variable(int index) {
        return new Expression(VAR, index, 0.0, null, null);
    }

    public static Expression constant(double value) {
        return new Expression(CONST, 0, value, null, null);
    }

    public static Expression op(int opCode, Expression left, Expression right) {
        return new Expression(OP, opCode, 0.0, left, right);
    }

    // ─── Evaluation ───────────────────────────────────────────────────────────

    /**
     * Evaluates the expression on a feature vector {@code x} of length 7.
     * Division by very small denominators returns 1.0 (protected division).
     */
    public double evaluate(double[] x) {
        switch (type) {
            case VAR:   return x[ivalue];
            case CONST: return dvalue;
            case OP: {
                double a = left.evaluate(x);
                double b = right.evaluate(x);
                switch (ivalue) {
                    case OP_ADD: return a + b;
                    case OP_SUB: return a - b;
                    case OP_MUL: return a * b;
                    case OP_DIV: return Math.abs(b) < 1e-9 ? 1.0 : a / b;
                }
            }
        }
        return 0.0; // unreachable
    }

    /** Returns the number of nodes in the expression tree (used for parsimony stats). */
    public int size() {
        if (type != OP) return 1;
        return 1 + left.size() + right.size();
    }

    // ─── Pretty-printing ──────────────────────────────────────────────────────

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        toString(sb);
        return sb.toString();
    }

    private void toString(StringBuilder sb) {
        switch (type) {
            case VAR:
                sb.append('x').append(ivalue + 1);
                return;
            case CONST:
                sb.append(String.format(java.util.Locale.US, "%.2f", dvalue));
                return;
            case OP:
                sb.append('(');
                left.toString(sb);
                sb.append(' ');
                switch (ivalue) {
                    case OP_ADD: sb.append('+'); break;
                    case OP_SUB: sb.append('-'); break;
                    case OP_MUL: sb.append('*'); break;
                    case OP_DIV: sb.append('/'); break;
                }
                sb.append(' ');
                right.toString(sb);
                sb.append(')');
        }
    }
}
