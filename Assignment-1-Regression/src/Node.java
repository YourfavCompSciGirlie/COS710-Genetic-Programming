import java.util.ArrayList;
import java.util.List;

/**
 * Represents a single node in a Genetic Programming expression tree.
 *
 * <p>A node is either:
 * <ul>
 *   <li><b>Terminal node</b> (arity = 0): a variable ("x1"–"x5") or an Ephemeral
 *       Random Constant (ERC), whose label is the numeric string of the constant.</li>
 *   <li><b>Function node</b> (arity = 2): one of the binary arithmetic operators
 *       "+", "-", "*", "/". Division is <em>protected</em> to ensure closure.</li>
 * </ul>
 *
 * <p>Each node keeps a reference to its parent to allow efficient upward tree
 * traversal during crossover and mutation depth calculations. Parent references
 * must be re-established after any structural operation by calling
 * {@link #setParents(Node, Node)}.
 *
 * <p>COS 710 Assignment 1 – Genetic Programming for Symbolic Regression.
 */
public class Node {

    /** The operator symbol ("+", "-", "*", "/"), variable name ("x1"–"x5"),
     *  or numeric ERC value as a formatted string (e.g. "2.4517"). */
    public String label;

    /** Number of children: 0 for terminals, 2 for binary operators. */
    public int arity;

    /** Ordered list of child nodes. Empty for terminal nodes. */
    public List<Node> children;

    /** Reference to the parent node; {@code null} if this is the tree root. */
    public Node parent;

    // ─── Constructor ──────────────────────────────────────────────────────────

    /**
     * Constructs a new Node with the given label and arity.
     * The children list is initialised as an empty ArrayList.
     *
     * @param label the operator symbol, variable name, or ERC string
     * @param arity 0 for terminals, 2 for binary function nodes
     */
    public Node(String label, int arity) {
        this.label    = label;
        this.arity    = arity;
        this.children = new ArrayList<>(arity);
        this.parent   = null;
    }

    // ─── Evaluation ───────────────────────────────────────────────────────────

    /**
     * Recursively evaluates this subtree given the input variable bindings.
     *
     * <p>Variable mapping: {@code inputs[0]} = x1 = Load(t−1),
     * {@code inputs[1]} = x2 = Load(t−2), …, {@code inputs[4]} = x5 = Load(t−5).
     *
     * <p>Protected division: {@code a / (|b| + 0.0001)} prevents divide-by-zero
     * and ensures the function set is closed over the reals.
     *
     * @param inputs array of input values corresponding to x1, x2, …, x5
     * @return the numeric result of evaluating this subtree
     * @throws IllegalStateException if an unrecognised operator label is encountered
     */
    public double evaluate(double[] inputs) {
        if (arity == 0) {
            // Terminal node: variable or ERC
            switch (label) {
                case "x1": return inputs[0];
                case "x2": return inputs[1];
                case "x3": return inputs[2];
                case "x4": return inputs[3];
                case "x5": return inputs[4];
                case "x6": return inputs[5];
                case "x7": return inputs[6];
                default:
                    // ERC: label is the numeric constant as a string
                    return Double.parseDouble(label);
            }
        }

        // Function node: evaluate both children first then apply operator
        double left  = children.get(0).evaluate(inputs);
        double right = children.get(1).evaluate(inputs);

        switch (label) {
            case "+": return left + right;
            case "-": return left - right;
            case "*": return left * right;
            case "/": return left / (Math.abs(right) + 0.0001); // protected division
            default:  throw new IllegalStateException("Unknown operator: " + label);
        }
    }

    // ─── Deep Copy ────────────────────────────────────────────────────────────

    /**
     * Creates a deep copy of this node and its entire subtree.
     *
     * <p><strong>Note:</strong> Parent references are <em>not</em> set by this method.
     * After copying, call {@link #setParents(Node, Node)} on the copied root to
     * re-establish all parent references throughout the subtree.
     *
     * @return a new Node that is a structural copy of this subtree
     */
    public Node copy() {
        Node clone = new Node(this.label, this.arity);
        for (Node child : this.children) {
            clone.children.add(child.copy());
        }
        return clone;
    }

    // ─── Structural Queries ───────────────────────────────────────────────────

    /**
     * Returns the depth of this subtree.
     * A leaf (terminal) node has depth 0; internal nodes have depth
     * {@code 1 + max(child depths)}.
     *
     * @return depth of this subtree
     */
    public int depth() {
        if (children.isEmpty()) return 0;
        int max = 0;
        for (Node child : children) {
            int d = child.depth();
            if (d > max) max = d;
        }
        return 1 + max;
    }

    /**
     * Returns the total number of nodes in this subtree, including this node.
     *
     * @return node count of this subtree
     */
    public int size() {
        int count = 1;
        for (Node child : children) {
            count += child.size();
        }
        return count;
    }

    /**
     * Returns {@code true} if this node is a terminal (leaf) node (arity == 0).
     *
     * @return {@code true} if terminal, {@code false} if function node
     */
    public boolean isTerminal() {
        return arity == 0;
    }

    /**
     * Performs a depth-first traversal of this subtree, appending every node
     * to the provided accumulator list. Used to build a flat list of all nodes
     * for random node selection during crossover and mutation.
     *
     * @param acc the list to which nodes are appended
     */
    public void getAllNodes(List<Node> acc) {
        acc.add(this);
        for (Node child : children) {
            child.getAllNodes(acc);
        }
    }

    // ─── Parent Management ────────────────────────────────────────────────────

    /**
     * Recursively sets the {@code parent} reference for every node in the subtree
     * rooted at {@code node}.
     *
     * <p>Must be called after any structural tree operation (copy, crossover,
     * mutation, trim) to keep parent references consistent.
     *
     * @param node   the root of the subtree whose parents are to be set
     * @param parent the intended parent of {@code node}; {@code null} if {@code node}
     *               is the tree root
     */
    public static void setParents(Node node, Node parent) {
        node.parent = parent;
        for (Node child : node.children) {
            setParents(child, node);
        }
    }

    // ─── String Representation ───────────────────────────────────────────────

    /**
     * Returns a fully-parenthesised infix string representation of this subtree.
     * Terminal nodes are displayed as their label string.
     * Example: {@code "(x1 + (x2 * x3))"}
     *
     * @return infix expression string
     */
    @Override
    public String toString() {
        if (arity == 0) return label;
        return "(" + children.get(0).toString()
                + " " + label + " "
                + children.get(1).toString() + ")";
    }
}
