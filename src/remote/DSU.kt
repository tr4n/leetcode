package remote

class UnionFind(n: Int) {
    private val parent = IntArray(n) { it }
    private val size = IntArray(n) { 1 }

    fun find(p: Int): Int {
        if (p == parent[p]) return p
        parent[p] = find(parent[p])
        return parent[p]
    }

    fun union(p: Int, q: Int) {
        val rootP = find(p)
        val rootQ = find(q)

        if (rootP != rootQ) {
            if (size[rootP] > size[rootQ]) {
                parent[rootQ] = rootP
                size[rootP] += size[rootQ]
            } else {
                parent[rootP] = rootQ
                size[rootQ] += size[rootP]
            }
        }
    }
}

fun findRedundantConnection(edges: Array<IntArray>): IntArray {
    val n = edges.size
    val uf = UnionFind(n + 1)
    for ((u, v) in edges) {
        if (uf.find(u) == uf.find(v)) {
            return intArrayOf(u, v)
        }
        uf.union(u, v)
    }
    return intArrayOf()
}

fun calcEquation(equations: List<List<String>>, values: DoubleArray, queries: List<List<String>>): DoubleArray {
    val name2Id = mutableMapOf<String, Int>()
    var n = 0

    for ((a, b) in equations) {
        name2Id.computeIfAbsent(a) { n++ }
        name2Id.computeIfAbsent(b) { n++ }
    }

    val parent = IntArray(n) { it }
    val weights = DoubleArray(n) { 1.0 }
    val size = IntArray(n) { 1 }

    fun find(p: Int): Pair<Int, Double> {
        if (p != parent[p]) {
            val (root, rootWeight) = find(parent[p])
            weights[p] *= rootWeight
            parent[p] = root
        }
        return parent[p] to weights[p]
    }

    fun union(p: Int, q: Int, value: Double) {
        val (rootP, weightP) = find(p)
        val (rootQ, weightQ) = find(q)
        // weightP = p / rootP => p = rootP * weightP
        // weightQ = q / rootQ => q = rootQ * weightQ
        // p/q = value,
        // rootP / rootQ = (p/q) / (weightP/weightQ) = value / (weightP/weightQ)
        if (rootP != rootQ) {
            val weightPQ = value / (weightP / weightQ)
            if (size[rootP] > size[rootQ]) {
                parent[rootQ] = rootP
                size[rootP] += size[rootQ]

                weights[rootQ] = 1.0 / weightPQ
            } else {
                parent[rootP] = rootQ
                size[rootQ] += size[rootP]
                weights[rootP] = weightPQ
            }
        }
    }

    for (i in equations.indices) {
        val (a, b) = equations[i]
        val idA = name2Id[a] ?: continue
        val idB = name2Id[b] ?: continue
        val value = values[i]
        union(idA, idB, value)
    }

    fun findResult(a: String, b: String): Double {
        val idA = name2Id[a] ?: return -1.0
        val idB = name2Id[b] ?: return -1.0
        val (rootA, weightA) = find(idA) // weightA = a/root
        val (rootB, weightB) = find(idB) // weightB = b/root
        if (rootA != rootB) return -1.0
        if (idA == idB) return 1.0

        return weightA / weightB
    }

    return DoubleArray(queries.size) {
        val (a, b) = queries[it]
        findResult(a, b)
    }
}

fun possibleBipartition(n: Int, dislikes: Array<IntArray>): Boolean {
    val parent = IntArray(n + 1) { it }
    // relation:
    // 1: is in the same group with root,
    // -1: is different from root
    val relation = IntArray(n + 1) { 1 }

    fun find(p: Int): Int {
        if (p == parent[p]) return p
        val root = find(parent[p])
        relation[p] = relation[p] * relation[parent[p]]
        parent[p] = root
        return root
    }

    fun uniteEnemies(a: Int, b: Int): Boolean {
        val rootA = find(a)
        val rootB = find(b)

        if (rootA == rootB) return relation[a] != relation[b]
        parent[rootB] = rootA
        // relation(a,b)= -1
        // relation(a, rootA) = relation[a]
        // relation(b, rootB) = relation[b]
        // => relation(rootB, rootA) = relation[a] * relation[b] / relation(a,b)
        relation[rootB] = -1 * relation[a] * relation[b]
        return true
    }

    for ((a, b) in dislikes) {
        if (!uniteEnemies(a, b)) return false
    }
    return true
}

fun isBipartite(graph: Array<IntArray>): Boolean {
    val n = graph.size
    val parent = IntArray(n) { it }
    val relation = IntArray(n) { 1 }

    fun find(u: Int): Int {
        if (u == parent[u]) return u
        val root = find(parent[u])
        relation[u] *= relation[parent[u]]
        parent[u] = root
        return root
    }

    fun separate(a: Int, b: Int): Boolean {
        val rootA = find(a)
        val rootB = find(b)
        if (rootA == rootB) return relation[a] != relation[b]
        parent[rootB] = rootA
        relation[rootB] = -1 * relation[a] * relation[b]
        return true
    }

    for(u in graph.indices) {
        for(v in graph[u]) {
            if (!separate(u, v)) return false
        }
    }

    return true
}

fun main() {

}