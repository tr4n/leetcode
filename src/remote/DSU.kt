package remote

import local.to2DIntArray
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

fun minimumEffortPath(heights: Array<IntArray>): Int {
    val rows = heights.size
    val cols = heights[0].size
    val dirX = intArrayOf(1, -1, 0, 0)
    val dirY = intArrayOf(0, 0, 1, -1)

    val pq = PriorityQueue<Triple<Int, Int, Int>>(compareBy { it.third })

    val effortTo = Array(rows) { IntArray(cols) { Int.MAX_VALUE } }


    effortTo[0][0] = 0
    pq.add(Triple(0, 0, 0))

    while (pq.isNotEmpty()) {
        val (row, col, currentEffort) = pq.poll()

        if (currentEffort > effortTo[row][col]) {
            continue
        }

        if (row == rows - 1 && col == cols - 1) {
            return currentEffort
        }

        for (i in 0 until 4) {
            val x = row + dirX[i]
            val y = col + dirY[i]

            if (x !in 0 until rows || y !in 0 until cols) {
                continue
            }

            val newEffort = maxOf(currentEffort, abs(heights[row][col] - heights[x][y]))

            if (newEffort < effortTo[x][y]) {
                effortTo[x][y] = newEffort
                pq.add(Triple(x, y, newEffort))
            }
        }
    }

    return 0
}

fun minCost(n: Int, edges: Array<IntArray>, k: Int): Int {
    val parent = IntArray(n) { it }
    val cost = IntArray(n)
    val size = IntArray(n) { 1 }
    var count = n

    fun find(u: Int): Int {
        if (u == parent[u]) return u
        val root = find(parent[u])
        parent[u] = root
        return root
    }

    fun union(a: Int, b: Int, weight: Int): Boolean {
        val rootA = find(a)
        val rootB = find(b)
        val newWeight = maxOf(cost[rootA], cost[rootB], weight)
        if (rootA == rootB) return false
        if (size[rootA] > size[rootB]) {
            size[rootA] += size[rootB]
            cost[rootA] = newWeight
            parent[rootB] = rootA
        } else {
            size[rootB] += size[rootA]
            cost[rootB] = newWeight
            parent[rootA] = rootB
        }
        count--
        return true
    }

    edges.sortBy { it[2] }
    val usedEdges = mutableListOf<IntArray>()

    for (edge in edges) {
        if (union(edge[0], edge[1], edge[2])) {
            usedEdges.add(edge)
        }
    }

    while (count < k && usedEdges.isNotEmpty()) {
        usedEdges.removeLast()
        count++
    }

    return usedEdges.lastOrNull()?.getOrNull(2) ?: 0
}

fun minMalwareSpread(graph: Array<IntArray>, initial: IntArray): Int {
    val n = graph.size

    val edges = Array(n) { mutableListOf<Int>() }
    for (i in 0 until n) {
        for (j in 0 until n) {
            if (graph[i][j] == 1) {
                edges[i].add(j)
                edges[j].add(i)
            }
        }
    }


    fun dsu(exceptNode: Int): Int {
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
        for (u in 0 until n) {
            if (u == exceptNode) continue
            for (v in edges[u]) {
                if (v == u || v == exceptNode) continue
                union(u, v)
            }
        }
        val groups = initial.filter { it != exceptNode }.map { find(it) }
            .distinct()
        //  println("Node $exceptNode: $groups, ${size.toList()}")
        val result = groups.sumOf { size[it] }
        //   .also { println("Remove $exceptNode: $it") }
        return result
    }

    return initial.sorted().minBy { dsu(it) }
}

fun removeStones(stones: Array<IntArray>): Int {
    val n = stones.size
    val parent = IntArray(n) { it }
    val size = IntArray(n) { 1 }
    var count = n

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
        count--
        return true
    }

    val firstRow = mutableMapOf<Int, Int>()
    val firstCol = mutableMapOf<Int, Int>()

    for (i in stones.indices) {
        val stone = stones[i]
        val firstRowStone = firstRow[stone[0]]
        if (firstRowStone == null) {
            firstRow[stone[0]] = i
        } else {
            union(firstRowStone, i)
        }
        val firstColStone = firstCol[stone[1]]
        if (firstColStone == null) {
            firstCol[stone[1]] = i
        } else {
            union(firstColStone, i)
        }
    }

    return n - count
}

fun largestComponentSize(nums: IntArray): Int {
    val n = nums.max()
    val present = BooleanArray(n + 1)
    val parent = IntArray(n + 1) { it }
    val size = IntArray(n + 1) { 1 }
    var count = n

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
        count--
        return true
    }

    for (num in nums) present[num] = true

    for (i in 2..n) {
        var multiple = i

        while (multiple <= n) {
            if (present[multiple]) {
                union(multiple, i)
            }
            multiple += i
        }
    }

    return nums.groupBy { find(it) }.maxOf { it.value.size }
}

fun reachableNodes(n: Int, edges: Array<IntArray>, restricted: IntArray): Int {
    val blockedNodes = restricted.toSet()
    val graph = Array(n) { mutableListOf<Int>() }
    for ((u, v) in edges) {
        if (u in blockedNodes || v in blockedNodes) continue
        graph[u].add(v)
        graph[v].add(u)
    }
    val visited = BooleanArray(n)
    var cnt = 0
    fun dfs(node: Int) {
        cnt++
        visited[node] = true

        for (v in graph[node]) {
            if (!visited[v]) dfs(v)
        }
    }
    dfs(0)
    return cnt
}

fun minScore(n: Int, roads: Array<IntArray>): Int {
    val parent = IntArray(n + 1) { it }
    val score = IntArray(n + 1) { Int.MAX_VALUE }
    val size = IntArray(n + 1) { 1 }

    fun find(u: Int): Int {
        if (u == parent[u]) return u
        val root = find(parent[u])
        parent[u] = root
        return root
    }

    fun union(a: Int, b: Int, distance: Int): Boolean {
        val rootA = find(a)
        val rootB = find(b)
        val newScore = minOf(score[rootA], score[rootB], distance)
        score[rootA] = newScore
        score[rootB] = newScore
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

    for (road in roads) {
        union(road[0], road[1], road[2])
    }

    return score[find(1)]
}

fun gcdSort(nums: IntArray): Boolean {
    val n = nums.max()
    val present = BooleanArray(n + 1)
    val parent = IntArray(n + 1) { it }
    val size = IntArray(n + 1) { 1 }
    var count = n

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
        count--
        return true
    }

    for (num in nums) present[num] = true

    for (i in 2..n) {
        var multiple = i

        while (multiple <= n) {
            if (present[multiple]) {
                union(multiple, i)
            }
            multiple += i
        }
    }

    val sortedNums = nums.sorted()

    for (i in nums.indices) {
        val originalNum = nums[i]
        val sortedNum = sortedNums[i]
        if (find(originalNum) != find(sortedNum)) {
            return false
        }
    }
    return true
}

fun friendRequests(n: Int, restrictions: Array<IntArray>, requests: Array<IntArray>): BooleanArray {

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

    val ans = BooleanArray(requests.size)

    for (i in requests.indices) {
        val x = requests[i][0]
        val y = requests[i][1]
        val rx = find(x)
        val ry = find(y)

        if (rx == ry) {
            ans[i] = true
            continue
        }

        var conflict = false
        for ((a, b) in restrictions) {
            val ra = find(a)
            val rb = find(b)
            if ((ra == rx && rb == ry) || (ra == ry && rb == rx)) {
                conflict = true
                break
            }
        }

        if (!conflict) {
            union(rx, ry)
            ans[i] = true
        } else {
            ans[i] = false
        }
    }
    return ans
}

fun countPairs(n: Int, edges: Array<IntArray>): Long {
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

    for ((a, b) in edges) union(a, b)
    var total = 0L
    for (i in 0 until n) {
        if (i != parent[i]) continue
        val x = size[i].toLong()
        total += x * (n.toLong() - x)
    }
    return total / 2L
}

fun lexicographicallySmallestArray(nums: IntArray, limit: Int): IntArray {
    val n = nums.size

    val result = IntArray(n) { nums[it] }
    nums.sort()
    var group = -1
    val groups = mutableListOf<MutableList<Int>>()
    val numToGroup = mutableMapOf<Int, Int>()
    var prev = Int.MIN_VALUE

    for (num in nums) {
        if (prev < 0 || num - prev > limit) {
            group++
            groups.add(mutableListOf(num))
        } else {
            groups[group].add(num)
        }
        prev = num
        numToGroup[num] = group
    }

    groups.onEach { it.sortDescending() }
    // println(result.toList())
    // println(groups)
    for (i in 0 until n) {
        val num = result[i]
        val groupId = numToGroup[num] ?: continue
        val smaller = groups[groupId].removeLastOrNull() ?: continue

        result[i] = smaller
    }
    return result
}

fun minimumCost(n: Int, edges: Array<IntArray>, query: Array<IntArray>): IntArray {
    val parent = IntArray(n) { it }
    val size = IntArray(n) { 1 }
    val cost = IntArray(n) { -1 }

    fun find(u: Int): Int {
        if (u == parent[u]) return u
        val root = find(parent[u])
        parent[u] = root
        return root
    }

    fun union(a: Int, b: Int, w: Int): Boolean {
        val rootA = find(a)
        val rootB = find(b)
        val newWeight = cost[rootA] and cost[rootB] and w
        cost[rootA] = newWeight
        cost[rootB] = newWeight

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

    for (edge in edges) union(edge[0], edge[1], edge[2])

    val result = IntArray(query.size) {
        val (start, end) = query[it]
        val rootA = find(start)
        val rootB = find(end)
        if (rootA != rootB) -1 else cost[rootA]
    }
    return result
}

fun magnificentSets(n: Int, edges: Array<IntArray>): Int {
    val parent = IntArray(n + 1) { it }
    val size = IntArray(n + 1) { 1 }

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
        if (size[rootA] >= size[rootB]) {
            size[rootA] += size[rootB]
            parent[rootB] = rootA
        } else {
            size[rootB] += size[rootA]
            parent[rootA] = rootB
        }
        return true
    }

    val graph = Array(n + 1) { mutableListOf<Int>() }
    for ((u, v) in edges) {
        graph[u].add(v)
        graph[v].add(u)
        union(u, v)
    }

    fun bfs(start: Int): Int {
        val ranks = IntArray(n + 1) { -1 }
        val queue = ArrayDeque<Int>()
        ranks[start] = 1
        queue.add(start)

        var maxDepth = 1
        while (queue.isNotEmpty()) {
            val u = queue.removeFirst()
            maxDepth = maxOf(maxDepth, ranks[u])

            for (v in graph[u]) {
                if (ranks[v] >= 0) {
                    if (abs(ranks[v] - ranks[u]) % 2 == 0) return Int.MIN_VALUE
                    continue
                }
                ranks[v] = ranks[u] + 1
                queue.add(v)
            }
        }

        return maxDepth
    }


    val groups = (1..n).groupBy { find(it) }
    var total = 0
    for (group in groups.values) {
        var maxDepth = -1
        for (node in group) {
            val depth = bfs(node)
            if (depth < 0) return -1
            maxDepth = maxOf(maxDepth, depth)
        }
        total += maxDepth
    }

    return total
}

fun maxPoints(grid: Array<IntArray>, queries: IntArray): IntArray {
    val m = grid.size
    val n = grid[0].size
    val dirX = intArrayOf(0, 0, 1, -1)
    val dirY = intArrayOf(-1, 1, 0, 0)

    val queryList = queries.withIndex().sortedBy { it.value }
    val result = IntArray(queryList.size)

    val pq = PriorityQueue<Pair<Int, Int>>(compareBy { grid[it.first][it.second] })
    pq.add(0 to 0)

    val visited = Array(m) { BooleanArray(n) }
    visited[0][0] = true

    var queryIndex = 0
    var cnt = 0

    while (queryIndex < queryList.size) {
        val query = queryList[queryIndex]
        val threshold = query.value
        while (pq.isNotEmpty() && grid[pq.peek().first][pq.peek().second] < threshold) {
            if (pq.isEmpty()) break
            val (row, col) = pq.peek()

            pq.poll()
            cnt++

            for (i in 0 until 4) {
                val x = row + dirX[i]
                val y = col + dirY[i]
                if (x !in 0 until m || y !in 0 until n) continue
                if (visited[x][y]) continue
                visited[x][y] = true
                pq.add(x to y)
            }
        }
        result[query.index] = cnt
        queryIndex++
    }

    return result
}

fun processQueries(c: Int, connections: Array<IntArray>, queries: Array<IntArray>): IntArray {
    val parent = IntArray(c + 1) { it }
    val size = IntArray(c + 1) { 1 }

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
        if (size[rootA] >= size[rootB]) {
            size[rootA] += size[rootB]
            parent[rootB] = rootA
        } else {
            size[rootB] += size[rootA]
            parent[rootA] = rootB
        }
        return true
    }

    for ((u, v) in connections) {
        union(u, v)
    }

    val map = mutableMapOf<Int, TreeSet<Int>>()

    for (i in 1..c) {
        val group = find(i)
        map.computeIfAbsent(group) { TreeSet() }.add(i)
    }

    val ans = mutableListOf<Int>()
    for (query in queries) {
        val x = query[1]
        val group = find(x)
        if (query[0] == 2) {
            //  println("Remove $x")
            map[group]?.remove(x)
            continue
        }
        // println("Query $x, ${map[group]}")
        val list = map[group] ?: emptyList()
        if (list.contains(x)) {
            ans.add(x)
            continue
        }
        ans.add(list.firstOrNull() ?: -1)
    }
    return ans.toIntArray()
}

fun canTraverseAllPairs(nums: IntArray): Boolean {
    val n = nums.max()

    val present = BooleanArray(n + 1)
    val parent = IntArray(n + 1) { it }
    val size = IntArray(n + 1) { 1 }
    var count = n

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
        count--
        return true
    }

    for (num in nums) present[num] = true

    if (present[1]) return nums.size == 1

    for (d in 2..n) {
        var multiple = d

        while (multiple <= n) {
            if (present[multiple]) {
                union(multiple, d)
            }
            multiple += d
        }
    }

    val group = find(nums[0])
    for (i in 1 until nums.size) {
        if (find(nums[i]) != group) return false
    }
    return true
}

fun minTime(n: Int, edges: Array<IntArray>, k: Int): Int {
    val parent = IntArray(n) { it }
    val size = IntArray(n) { 1 }
    var count = n

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
        if (size[rootA] >= size[rootB]) {
            size[rootA] += size[rootB]
            parent[rootB] = rootA
        } else {
            size[rootB] += size[rootA]
            parent[rootA] = rootB
        }
        count--
        return true
    }

    edges.sortByDescending { it[2] }
    val m = edges.size
    val counts = IntArray(m)
    for (i in 0 until m) {
        val (u, v, time) = edges[i]
        union(u, v)
        counts[i] = count
    }

    if (count >= k) return 0
    var lo = 0
    var hi = m - 1
    var index = -1

    while (lo <= hi) {
        val mid = (lo + hi) / 2
        val cnt = counts[mid]
        if (cnt >= k) {
            index = mid
            lo = mid + 1
        } else {
            hi = mid - 1
        }
    }
    //  println(counts.toList())
    return if (index + 1 >= m) 0 else edges[index + 1][2]
}

fun countIslands(grid: Array<IntArray>, k: Int): Int {
    val m = grid.size
    val n = grid[0].size

    fun fill(row: Int, col: Int): Long {
        if (row !in 0 until m || col !in 0 until n) return 0
        if (grid[row][col] <= 0) return 0
        var result = grid[row][col].toLong()
        grid[row][col] = -1

        result += fill(row + 1, col)
        result += fill(row - 1, col)
        result += fill(row, col + 1)
        result += fill(row, col - 1)
        return result
    }

    var cnt = 0
    for (i in 0 until m) {
        for (j in 0 until n) {
            if (grid[i][j] <= 0) continue
            val totalValue = fill(i, j)
            if (totalValue % k == 0L) cnt++
        }
    }
    return cnt
}

fun pathExistenceQueries(n: Int, nums: IntArray, maxDiff: Int, queries: Array<IntArray>): BooleanArray {
    var group = -1
    val numToGroup = IntArray(n) { -1 }
    var prev = Int.MIN_VALUE

    for (i in nums.indices) {
        val num = nums[i]
        if (prev < 0 || num - prev > maxDiff) {
            group++
        }
        prev = num
        numToGroup[i] = group
    }

    return BooleanArray(queries.size) {
        val (u, v) = queries[it]
        numToGroup[u] == numToGroup[v]
    }
}

fun numberOfComponents(properties: Array<IntArray>, k: Int): Int {
    val n = properties.size
    val props = properties.map { it.toSet() }
    val parent = IntArray(n) { it }
    val size = IntArray(n) { 1 }
    var count = n

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
        if (size[rootA] >= size[rootB]) {
            size[rootA] += size[rootB]
            parent[rootB] = rootA
        } else {
            size[rootB] += size[rootA]
            parent[rootA] = rootB
        }
        count--
        return true
    }

    for (i in 0 until n - 1) {
        for (j in i + 1 until n) {
            val r1 = find(i)
            val r2 = find(j)
            if (r1 == r2) continue
            val intersect = props[i].intersect(props[j]).size
            if (intersect >= k) union(i, j)
        }
    }
    return count
}

fun smallestEquivalentString(s1: String, s2: String, baseStr: String): String {
    val n = 123
    val parent = IntArray(n) { it }
    val size = IntArray(n) { 1 }
    val smallest = IntArray(n) { if (it in 'a'.code..'z'.code) it else Int.MAX_VALUE }
    var count = n

    fun find(u: Int): Int {
        if (u == parent[u]) return u
        val root = find(parent[u])
        parent[u] = root
        return root
    }

    fun union(a: Int, b: Int): Boolean {
        val rootA = find(a)
        val rootB = find(b)
        val min = minOf(smallest[rootA], smallest[rootB])
        if (rootA == rootB) {
            smallest[rootA] = min
            smallest[rootB] = min
            return false
        }
        if (size[rootA] >= size[rootB]) {
            size[rootA] += size[rootB]
            parent[rootB] = rootA
        } else {
            size[rootB] += size[rootA]
            parent[rootA] = rootB
        }
        smallest[rootA] = min
        smallest[rootB] = min
        count--
        return true
    }

    for (i in 0 until s1.length) {
        val a = s1[i].code
        val b = s2[i].code
        union(a, b)
    }

    val builder = StringBuilder()
    for (c in baseStr) {
        val ch = smallest[find(c.code)].toChar()
        builder.append(ch)
    }
    return builder.toString()
}

fun hasValidPath(grid: Array<IntArray>): Boolean {
    val m = grid.size
    val n = grid[0].size

    fun getNext(r: Int, c: Int): List<Pair<Int, Int>> {
        val dirs = mapOf(
            1 to listOf(0 to -1, 0 to 1),
            2 to listOf(-1 to 0, 1 to 0),
            3 to listOf(0 to -1, 1 to 0),
            4 to listOf(0 to 1, 1 to 0),
            5 to listOf(0 to -1, 0 to -1),
            6 to listOf(0 to 1, 0 to -1)
        )

        fun isConnectedBack(d: Pair<Int, Int>, nr: Int, nc: Int): Boolean {
            val opposite = (-d.first to -d.second)
            val nextType = grid[nr][nc]
            val nextDirs = dirs[nextType] ?: emptyList()
            return opposite in nextDirs
        }

        val type = grid[r][c]
        val res = mutableListOf<Pair<Int, Int>>()
        for ((dr, dc) in dirs[type] ?: emptyList()) {
            val nr = r + dr
            val nc = c + dc
            if (nr in 0 until m && nc in 0 until n) {
                if (isConnectedBack(dr to dc, nr, nc)) {
                    res.add(nr to nc)
                }
            }
        }
        return res
    }

    val memo = Array(m) { BooleanArray(n) }
    memo[m - 1][n - 1] = true
    fun dfs(row: Int, col: Int): Boolean {
        if (row == m - 1 && col == n - 1) return true
        if (memo[row][col]) return true

        val nextList = getNext(row, col)
        var result = false
        for ((x, y) in nextList) {
            if (dfs(x, y)) {
                result = true
                break
            }
        }
        memo[row][col] = result
        return result
    }
    return dfs(0, 0)
}

fun findAllPeople(n: Int, meetings: Array<IntArray>, firstPerson: Int): List<Int> {
    val parent = IntArray(n) { it }
    val size = IntArray(n) { 1 }

    var count = n

    fun find(u: Int): Int {
        if (u == parent[u]) return u
        val root = find(parent[u])
        parent[u] = root
        return root
    }

    fun union(a: Int, b: Int): Boolean {
        val rootA = find(a)
        val rootB = find(b)

        if (rootA == rootB) {
            return false
        }
        if (size[rootA] >= size[rootB]) {
            size[rootA] += size[rootB]
            parent[rootB] = rootA
        } else {
            size[rootB] += size[rootA]
            parent[rootA] = rootB
        }
        count--
        return true
    }

    meetings.groupBy { it[3] }
    union(0, firstPerson)
    val edges = meetings.withIndex().sortedWith(compareBy<IndexedValue<IntArray>> { it.value[2] }.thenBy { it.index })
    println(edges.map { it.value.toList() })
    for (meeting in edges) {
        val (a, b) = meeting.value
        val secret = find(0)
        if (find(a) == secret || find(b) == secret) union(a, b)
    }
    val group = find(0)
    return (0 until n).filter { find(it) == group }
}

fun main() {
    println(
        findAllPeople(5, "[[1,4,3],[0,4,3]]".to2DIntArray(), 3)
    )

}