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
 * <p>For Structure-Based GP, this class provides
 * {@link #structuralSimilarity(Tree)}, which normalises the raw prefix-match
 * count from {@link Node#prefixMatchCount(Node, Node)} into a [0, 1] similarity
 * score. A score of 1.0 means the two trees are structurally identical; 0.0
 * means they share no common prefix from the root.
 *
 * <p>COS 710 Assignment 2 – Structure-Based Genetic Programming for Regression.
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

    // ─── Structural Similarity (SBGP) ────────────────────────────────────────

    /**
     * Computes the structural similarity between this tree and another, using the
     * top-down prefix node matching approach described in the COS 710 lecture
     * slides on Structure-Based GP.
     *
     * <p>The raw prefix match count (see {@link Node#prefixMatchCount(Node, Node)})
     * is normalised by the size of the larger tree so that the result falls in
     * the range [0.0, 1.0]:
     * <pre>
     *   similarity = prefixMatchCount(root1, root2) / max(size(T1), size(T2))
     * </pre>
     *
     * <p>Boundary cases:
     * <ul>
     *   <li>Two structurally identical trees return 1.0.</li>
     *   <li>Two trees with different root operator/terminal labels return 0.0.</li>
     * </ul>
     *
     * @param other the tree to compare against
     * @return normalised structural similarity in [0.0, 1.0]
     */
    public double structuralSimilarity(Tree other) {
        int matchCount = Node.prefixMatchCount(this.root, other.root);
        int maxSize    = Math.max(this.size(), other.size());
        return maxSize == 0 ? 1.0 : (double) matchCount / maxSize;
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
     * @param node         the current node being examined
     * @param currentDepth the depth of {@code node} measured from the tree root
     * @param maxDepth     the maximum allowed tree depth
     * @param terminals    the variable terminal labels used when creating replacement nodes
     * @param rng          random number generator for replacement terminal selection
     */
    public static void trimToDepth(Node node, int currentDepth, int maxDepth,
                                   String[] terminals, Random rng) {
        if (!node.isTerminal() && currentDepth >= maxDepth) {
            Node replacement = makeTerminalNode(terminals, rng);
            node.label = replacement.label;
            node.arity = 0;
            node.children.clear();
            return;
        }
        for (Node child : node.children) {
            trimToDepth(child, currentDepth + 1, maxDepth, terminals, rng);
        }
    }

    // ─── Terminal Factory ─────────────────────────────────────────────────────

    /**
     * Creates a random terminal node.
     *
     * <p>With 30% probability an Ephemeral Random Constant (ERC) is created:
     * a random double in {@code [−5.0, 5.0]} formatted to four decimal places.
     * Otherwise a random variable terminal from {@code terminals} is chosen.
     *
     * @param terminals array of variable terminal labels (e.g. {"x1",…,"x7"})
     * @param rng       random number generator
     * @return a new terminal {@link Node}
     */
    static Node makeTerminalNode(String[] terminals, Random rng) {
        if (rng.nextDouble() < 0.30) {
            double value = rng.nextDouble() * 10.0 - 5.0;
            return new Node(String.format(Locale.US, "%.4f", value), 0);
        }
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
