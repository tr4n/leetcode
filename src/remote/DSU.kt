package remote

import java.util.*
import kotlin.math.abs

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

    for (u in graph.indices) {
        for (v in graph[u]) {
            if (!separate(u, v)) return false
        }
    }

    return true
}

fun numSimilarGroups(strs: Array<String>): Int {
    fun areSimilar(s1: String, s2: String): Boolean {
        val diffIndices = mutableListOf<Int>()
        for (i in s1.indices) {
            if (s1[i] != s2[i]) {
                diffIndices.add(i)
            }
        }

        return when (diffIndices.size) {
            0 -> true
            2 -> {
                val i = diffIndices[0]
                val j = diffIndices[1]
                s1[i] == s2[j] && s1[j] == s2[i]
            }

            else -> false
        }
    }

    val n = strs.size
    val parent = IntArray(n) { it }

    fun find(u: Int): Int {
        if (u == parent[u]) return u
        val root = find(parent[u])
        parent[u] = root
        return root
    }

    fun union(a: Int, b: Int) {
        val rootA = find(a)
        val rootB = find(b)
        if (rootA == rootB) return
        parent[rootB] = rootA
    }

    for (i in 0 until n - 1) {
        for (j in i + 1 until n) {
            val s1 = strs[i]
            val s2 = strs[j]
            if (areSimilar(s1, s2)) {
                union(i, j)
            }
        }
    }
    return parent.indices.count { it == parent[it] }
}

fun countCompleteComponents(n: Int, edges: Array<IntArray>): Int {
    val parent = IntArray(n) { it }
    val nodeCount = IntArray(n) { 1 }
    val edgeCount = IntArray(n)

    fun find(u: Int): Int {
        if (u == parent[u]) return u
        val root = find(parent[u])
        parent[u] = root
        return root
    }

    fun union(a: Int, b: Int) {
        val rootA = find(a)
        val rootB = find(b)
        if (rootA == rootB) {
            edgeCount[rootA]++
            return
        }
        if (nodeCount[rootA] > nodeCount[rootB]) {
            parent[rootB] = rootA
            nodeCount[rootA] += nodeCount[rootB]
            edgeCount[rootA] += edgeCount[rootB] + 1
        } else {
            parent[rootA] = rootB
            nodeCount[rootB] += nodeCount[rootA]
            edgeCount[rootB] += edgeCount[rootA] + 1
        }
    }

    for ((a, b) in edges) {
        union(a, b)
    }

    var cnt = 0
    for (i in 0 until n) {
        if (i != parent[i]) continue
        val m = nodeCount[i]
        val e = edgeCount[i]
        if (2 * e == m * (m - 1)) cnt++
    }
    return cnt
}

fun accountsMerge(accounts: List<List<String>>): List<List<String>> {

    val email2Id = mutableMapOf<String, Int>()
    val names = mutableListOf<String>()
    val id2Email = mutableListOf<String>()

    var n = 0

    for (list in accounts) {
        val name = list.first()
        for (i in 1 until list.size) {
            val email = list[i]
            if (email2Id[email] != null) continue
            id2Email.add(email)
            names.add(name)
            email2Id[email] = n++
        }
    }

    val parent = IntArray(n) { it }
    val size = IntArray(n) { 1 }

    fun find(u: Int): Int {
        if (u == parent[u]) return u
        val root = find(parent[u])
        parent[u] = root
        return root
    }

    fun union(a: Int, b: Int) {
        val rootA = find(a)
        val rootB = find(b)
        if (rootA == rootB) return
        if (size[rootA] > size[rootB]) {
            parent[rootB] = rootA
            size[rootA] += size[rootB]
        } else {
            parent[rootA] = rootB
            size[rootB] += size[rootA]
        }
    }

    for (list in accounts) {
        val firstEmail = list[1]
        val firstId = email2Id[firstEmail] ?: continue

        for (i in 2 until list.size) {
            val email = list[i]
            val emailId = email2Id[email] ?: continue
            union(emailId, firstId)
        }
    }

    return parent.indices.groupBy { find(it) }.map { (root, ids) ->
        val name = names[root]
        val line = mutableListOf(name)
        line.addAll(ids.map { id2Email[it] }.sorted())
        line
    }
}

fun smallestStringWithSwaps(s: String, pairs: List<List<Int>>): String {
    val n = s.length
    val parent = IntArray(n) { it }
    val size = IntArray(n) { 1 }

    fun find(u: Int): Int {
        if (u == parent[u]) return u
        val root = find(parent[u])
        parent[u] = root
        return root
    }

    fun union(a: Int, b: Int) {
        val rootA = find(a)
        val rootB = find(b)
        if (rootA == rootB) return
        if (size[rootA] > size[rootB]) {
            parent[rootB] = rootA
            size[rootA] += size[rootB]
        } else {
            parent[rootA] = rootB
            size[rootB] += size[rootA]
        }
    }

    for ((a, b) in pairs) {
        union(a, b)
    }

    val components = mutableMapOf<Int, PriorityQueue<Char>>()
    for (i in 0 until n) {
        val root = find(i)
        components.getOrPut(root) { PriorityQueue() }.add(s[i])
    }

    val result = CharArray(n)
    for (i in 0 until n) {
        val root = find(i)
        result[i] = components[root]?.poll() ?: '_'
    }

    return String(result)
}

fun countServers(grid: Array<IntArray>): Int {

    val rows = grid.size
    val cols = grid[0].size
    val cellToId = Array(rows) { IntArray(cols) { -1 } }
    var n = 0

    for (i in 0 until rows) {
        for (j in 0 until cols) {
            if (grid[i][j] == 0) continue
            if (cellToId[i][j] >= 0) continue
            cellToId[i][j] = n++
        }
    }

    val parent = IntArray(n) { it }
    val size = IntArray(n) { 1 }

    fun find(u: Int): Int {
        if (u == parent[u]) return u
        val root = find(parent[u])
        parent[u] = root
        return root
    }

    fun union(a: Int, b: Int) {
        val rootA = find(a)
        val rootB = find(b)
        if (rootA == rootB) return
        if (size[rootA] > size[rootB]) {
            parent[rootB] = rootA
            size[rootA] += size[rootB]
        } else {
            parent[rootA] = rootB
            size[rootB] += size[rootA]
        }
    }

    val firstRow = IntArray(rows) { -1 }
    val firstCol = IntArray(cols) { -1 }

    for (i in 0 until rows) {
        for (j in 0 until cols) {
            val id = cellToId[i][j]
            if (id < 0) continue

            if (firstRow[i] < 0) {
                firstRow[i] = id
            } else union(firstRow[i], id)

            if (firstCol[j] < 0) {
                firstCol[j] = id
            } else union(firstCol[j], id)
        }
    }

    return (0 until n).sumOf {
        if (it == parent[it] && size[it] > 1) size[it] else 0
    }
}

fun makeConnected(n: Int, connections: Array<IntArray>): Int {
    val parent = IntArray(n) { it }
    val size = IntArray(n) { 1 }

    fun find(u: Int): Int {
        if (u == parent[u]) return u
        val root = find(parent[u])
        parent[u] = root
        return root
    }

    fun union(a: Int, b: Int): Boolean {
        val rootA = find(a)
        val rootB = find(b)
        if (rootA == rootB) return false
        if (size[rootA] > size[rootB]) {
            parent[rootB] = rootA
            size[rootA] += size[rootB]
        } else {
            parent[rootA] = rootB
            size[rootB] += size[rootA]
        }
        return true
    }

    var capCount = 0
    for ((a, b) in connections) {
        if (!union(a, b)) capCount++
    }

    var groupCount = 0
    for (i in 0 until n) {
        if (parent[i] == i) groupCount++
    }
    return if (capCount >= groupCount - 1) groupCount - 1 else -1
}

fun validateBinaryTreeNodes(n: Int, leftChild: IntArray, rightChild: IntArray): Boolean {
    val parent = IntArray(n) { -1 }
    val ancestor = IntArray(n) { it }
    val size = IntArray(n) { 1 }

    fun find(u: Int): Int {
        if (u == ancestor[u]) return u
        val root = find(ancestor[u])
        ancestor[u] = root
        return root
    }

    fun union(a: Int, b: Int): Boolean {
        val rootA = find(a)
        val rootB = find(b)
        if (rootA == rootB) return false
        if (size[rootA] > size[rootB]) {
            ancestor[rootB] = rootA
            size[rootA] += size[rootB]
        } else {
            ancestor[rootA] = rootB
            size[rootB] += size[rootA]
        }
        return true
    }


    for (i in 0 until n) {
        val left = leftChild[i]
        val right = rightChild[i]
        if (left >= 0) {
            if (parent[left] >= 0 || parent[i] == left) return false
            parent[left] = i
            if (!union(i, left)) return false
        }
        if (right >= 0) {
            if (parent[right] >= 0 || parent[i] == right) return false
            parent[right] = i
            if (!union(i, right)) return false
        }
    }

    return (0 until n).count { ancestor[it] == it } == 1
}

fun minCostConnectPoints(points: Array<IntArray>): Int {
    class Edge(val a: Int, val b: Int) {
        val dist = abs(points[a][0] - points[b][0]) + abs(points[a][1] - points[b][1])
    }

    val n = points.size
    val parent = IntArray(n) { it }
    val size = IntArray(n) { 1 }

    fun find(u: Int): Int {
        if (u == parent[u]) return u
        val root = find(parent[u])
        parent[u] = root
        return root
    }

    fun union(a: Int, b: Int): Boolean {
        val rootA = find(a)
        val rootB = find(b)
        if (rootA == rootB) return false
        if (size[rootA] > size[rootB]) {
            size[rootA] += size[rootB]
            parent[rootB] = rootA
        } else {
            size[rootB] += size[rootA]
            parent[rootA] = rootB
        }
        return true
    }

    val edges = mutableListOf<Edge>()

    for (i in 0 until n - 1) {
        for (j in i + 1 until n) {
            edges.add(Edge(i, j))
        }
    }
    edges.sortBy { it.dist }

    var edgeCount = 0
    var edgeId = 0
    var cost = 0
    while (edgeCount < n - 1) {
        val edge = edges[edgeId++]
        if (union(edge.a, edge.b)) {
            cost += edge.dist
            edgeCount++
        }
    }
    return cost
}

fun main() {
    println(
        numSimilarGroups(
            arrayOf(
                "kccomwcgcs",
                "socgcmcwkc",
                "sgckwcmcoc",
                "coswcmcgkc",
                "cowkccmsgc",
                "cosgmccwkc",
                "sgmkwcccoc",
                "coswmccgkc",
                "kowcccmsgc",
                "kgcomwcccs"
            )
        )
    )

}