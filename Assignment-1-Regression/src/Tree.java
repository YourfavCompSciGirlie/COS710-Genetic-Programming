import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Random;

/**
 * Represents a single Genetic Programming individual: a full expression tree.
 *
 * <p>Each {@code Tree} wraps a root {@link Node} and provides the interface used
 * by the GP engine for evaluation, copying, random node selection, and bloat
 * control. Every structural operation that modifies the tree calls
 * {@link Node#setParents(Node, Node)} to keep parent references valid.
 *
 * <p>COS 710 Assignment 1 – Genetic Programming for Symbolic Regression.
 */
public class Tree {

    /** The root node of the expression tree. */
    public Node root;

    // ─── Constructor ──────────────────────────────────────────────────────────

    /**
     * Constructs a Tree with the given root node and immediately establishes
     * all parent references throughout the subtree.
     *
     * @param root the root node of this expression tree (must not be {@code null})
     */
    public Tree(Node root) {
        this.root = root;
        Node.setParents(root, null);
    }

    // ─── Evaluation ───────────────────────────────────────────────────────────

    /**
     * Evaluates this tree expression against a set of input variable values.
     *
     * @param inputs array of doubles where {@code inputs[0]} = x1 = Load(t−1),
     *               {@code inputs[1]} = x2 = Load(t−2), etc.
     * @return the computed value of the expression tree
     */
    public double evaluate(double[] inputs) {
        return root.evaluate(inputs);
    }

    // ─── Deep Copy ────────────────────────────────────────────────────────────

    /**
     * Creates a deep copy of this tree with all parent references re-established.
     * The original tree is not modified.
     *
     * @return a new {@code Tree} that is a structural and logical copy of this one
     */
    public Tree copy() {
        Node rootCopy = root.copy();
        Node.setParents(rootCopy, null);
        return new Tree(rootCopy);
    }

    // ─── Node Access ──────────────────────────────────────────────────────────

    /**
     * Returns a flat list of every node in the tree via a depth-first traversal.
     * Used to select random crossover and mutation points.
     *
     * @return list of all nodes in depth-first order
     */
    public List<Node> getAllNodes() {
        List<Node> nodes = new ArrayList<>();
        root.getAllNodes(nodes);
        return nodes;
    }

    /**
     * Selects and returns a uniformly random node from the tree.
     *
     * @param rng the random number generator to use for selection
     * @return a randomly chosen {@link Node} within this tree
     */
    public Node getRandomNode(Random rng) {
        List<Node> nodes = getAllNodes();
        return nodes.get(rng.nextInt(nodes.size()));
    }

    // ─── Structural Queries ───────────────────────────────────────────────────

    /**
     * Returns the depth of this tree (depth of root's subtree).
     * A tree consisting of only a root terminal has depth 0.
     *
     * @return tree depth
     */
    public int depth() {
        return root.depth();
    }

    /**
     * Returns the total number of nodes in this tree.
     *
     * @return node count
     */
    public int size() {
        return root.size();
    }

    // ─── Bloat Control ────────────────────────────────────────────────────────

    /**
     * Enforces a maximum tree depth by trimming any subtree that exceeds
     * {@code maxDepth}. Operates in-place on the given node.
     *
     * <p>If a function node is found at {@code currentDepth >= maxDepth}, it is
     * converted in-place to a random terminal (its children are discarded).
     * This ensures the tree depth invariant {@code depth ≤ maxDepth} holds after
     * every crossover and mutation operation.
     *
     * <p>After calling this method, the caller should invoke
     * {@link Node#setParents(Node, Node)} on the tree root to update all parent
     * references that may have been modified.
     *
     * @param node         the current node being examined
     * @param currentDepth the depth of {@code node} measured from the tree root
     *                     (root = depth 0)
     * @param maxDepth     the maximum allowed tree depth
     * @param terminals    the variable terminal labels to use when creating
     *                     replacement terminal nodes (e.g. {"x1",…,"x5"})
     * @param rng          random number generator for replacement terminal selection
     */
    public static void trimToDepth(Node node, int currentDepth, int maxDepth,
                                   String[] terminals, Random rng) {
        // If this function node exceeds the depth limit, convert it to a terminal
        if (!node.isTerminal() && currentDepth >= maxDepth) {
            Node replacement = makeTerminalNode(terminals, rng);
            node.label = replacement.label;
            node.arity = 0;
            node.children.clear();
            return; // now a leaf – no children to recurse into
        }
        // Recursively process all children
        for (Node child : node.children) {
            trimToDepth(child, currentDepth + 1, maxDepth, terminals, rng);
        }
    }

    // ─── Terminal Factory ─────────────────────────────────────────────────────

    /**
     * Creates a random terminal node.
     *
     * <p>With 30% probability, an Ephemeral Random Constant (ERC) is created:
     * a random double in {@code [−5.0, 5.0]} formatted to four decimal places.
     * Otherwise, a random variable terminal from {@code terminals} is chosen.
     *
     * @param terminals array of variable terminal labels (e.g. {"x1","x2","x3","x4","x5"})
     * @param rng       random number generator
     * @return a new terminal {@link Node}
     */
    static Node makeTerminalNode(String[] terminals, Random rng) {
        if (rng.nextDouble() < 0.30) {
            // Ephemeral Random Constant: uniform in [-5.0, 5.0]
            double value = rng.nextDouble() * 10.0 - 5.0;
            // Always use Locale.US so the decimal separator is a dot; otherwise
            // Double.parseDouble() will fail on locales that use commas (e.g. French).
            return new Node(String.format(Locale.US, "%.4f", value), 0);
        }
        // Variable terminal: randomly chosen from the terminal set
        String label = terminals[rng.nextInt(terminals.length)];
        return new Node(label, 0);
    }

    // ─── String Representation ───────────────────────────────────────────────

    /**
     * Returns the infix string representation of this tree's expression.
     * Delegates to the root node's {@link Node#toString()} method.
     *
     * @return fully-parenthesised infix expression string
     */
    @Override
    public String toString() {
        return root.toString();
    }
}
