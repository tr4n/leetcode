package remote

import java.util.*
import kotlin.math.abs

fun canVisitAllRooms(rooms: List<List<Int>>): Boolean {
    val n = rooms.size
    val visited = BooleanArray(n)

    fun visit(u: Int) {
        visited[u] = true

        for (v in rooms[u]) {
            if (!visited[v]) {
                visit(v)
            }
        }
    }

    visit(0)

    return visited.all { it }

}

fun shortestAlternatingPaths(n: Int, redEdges: Array<IntArray>, blueEdges: Array<IntArray>): IntArray {

    val redGraph = Array(n) { mutableListOf<Int>() }
    val blueGraph = Array(n) { mutableListOf<Int>() }
    for (edge in redEdges) redGraph[edge[0]].add(edge[1])
    for (edge in blueEdges) blueGraph[edge[0]].add(edge[1])

    val visited = mutableSetOf<Pair<Int, Boolean>>()
    val d = Array(n) { IntArray(2) { Int.MAX_VALUE } }
    d[0][0] = 0
    d[0][1] = 0

    fun visit(u: Int, fromRed: Boolean) {
        visited.add(u to fromRed)

        val nextNodes = if (fromRed) blueGraph[u] else redGraph[u]
        if (nextNodes.isEmpty()) {
            return
        }
        for (v in nextNodes) {
            if (v to !fromRed !in visited) {
                if (fromRed) {
                    d[v][1] = minOf(d[v][1], d[u][0] + 1)
                } else {
                    d[v][0] = minOf(d[v][0], d[u][1] + 1)
                }
                visit(v, !fromRed)
                visited.remove(v to !fromRed)
            }
        }
    }

    fun bfs(startNode: Int, startColor: Boolean) {
        val queue = ArrayDeque<Pair<Int, Boolean>>()
        queue.addLast(startNode to startColor)

        while (queue.isNotEmpty()) {
            val (u, fromRed) = queue.removeFirst()
            val nextNodes = if (fromRed) blueGraph[u] else redGraph[u]
            if (nextNodes.isEmpty()) {
                continue
            }
            for (v in nextNodes) {
                if (v to !fromRed !in visited) {
                    if (fromRed) {
                        d[v][1] = minOf(d[v][1], d[u][0] + 1)
                    } else {
                        d[v][0] = minOf(d[v][0], d[u][1] + 1)
                    }
                    queue.add(v to !fromRed)
                    //visit(v, !fromRed)
                    visited.remove(v to !fromRed)
                }
            }
        }
    }


    //  visit(0, true)
    bfs(0, true)
    visited.clear()
    bfs(0, false)
    // println(d.map { it[0] })
    //   visit(0, false)
    //  println(d.map { it[1] })

    return IntArray(n) {
        val value = d[it].min()
        if (value == Int.MAX_VALUE) -1 else value
    }
}

fun numOfMinutes(n: Int, headID: Int, manager: IntArray, informTime: IntArray): Int {
    val graph = Array(n) { mutableListOf<Int>() }
    val visited = BooleanArray(n)

    for (i in 0 until n) {
        val managerId = manager[i]
        if (managerId !in 0 until n) continue
        graph[managerId].add(i)
    }
    val queue = ArrayDeque<Pair<Int, Int>>()
    queue.addLast(headID to 0)
    var totalTime = 0
    while (queue.isNotEmpty()) {
        val (u, time) = queue.removeFirst()
        visited[u] = true

        if (graph[u].isEmpty()) {
            totalTime = maxOf(totalTime, time)
        }
        val nextTime = time + informTime[u]

        for (v in graph[u]) {
            if (!visited[v]) {
                queue.add(v to nextTime)
                //visit(v, !fromRed)
                // visited.remove(v to !fromRed)
            }
        }
    }
    return totalTime
}

fun maxDepth(root: TreeNode?): Int {
    if (root == null) return 0
    val left = maxDepth(root.left)
    val right = maxDepth(root.right)
    return 1 + maxOf(left, right)
}

fun minReorder(n: Int, connections: Array<IntArray>): Int {
    val graph = Array(n) { mutableListOf<Int>() }
    //   val edges = mutableSetOf<Pair<Int, Int>>()
    val originSet = mutableSetOf<Pair<Int, Int>>()
    for ((u, v) in connections) {
        graph[v].add(u)
        graph[u].add(v)
        originSet.add(u to v)
    }
    val visited = BooleanArray(n)
    var cnt = 0
    fun visit(u: Int) {
        if (u == 0) {
            return
        }

        visited[u] = true


        for (v in graph[u]) {
            if (!visited[v]) {
                if (u to v in originSet) {
                    cnt++
                }
                //   edges.add(u to v)
                visit(v)
            }
        }
    }

    //   println(originSet)
    //  println(edges)
//    for(i in (n-1) downTo 1) {
//        if(!visited[i]) {
//            visit(i)
//        }
//    }
    visit(0)
    return cnt
}

fun allPathsSourceTarget(graph: Array<IntArray>): List<List<Int>> {
    val n = graph.size
    val visited = BooleanArray(n)
    val result = mutableListOf<List<Int>>()
    val path = mutableListOf<Int>()
    fun visit(u: Int) {
        path.add(u)
        if (u == n - 1) {
            result.add(path.toList())
            return
        }

        visited[u] = true


        for (v in graph[u]) {
            if (!visited[v]) {
                visit(v)
                path.removeLast()
                visited[v] = false
            }
        }
    }
    visit(0)
    return result
}

fun countPaths(n: Int, roads: Array<IntArray>): Int {
    val mod = 1_000_000_007
    val graph = Array(n) { mutableListOf<Pair<Int, Int>>() }

    for ((u, v, time) in roads) {
        graph[u].add(v to time)
        graph[v].add(u to time)
    }

    var minTotalTime = Double.MAX_VALUE
    val d = DoubleArray(n) { Double.MAX_VALUE }
    d[0] = 0.0
    val dp = LongArray(n)
    dp[0] = 1
    val queue = PriorityQueue<Pair<Int, Double>>(compareBy({ it.second }, { -it.first }))
    queue.add(0 to 0.0)
    while (queue.isNotEmpty()) {
        val (u, uTime) = queue.poll()
        if (uTime != d[u]) continue
        //  println("Visit $u ${d[u]}")
        if (u == n - 1) {
            when {
                uTime < minTotalTime -> {
                    minTotalTime = uTime
                }

                uTime == minTotalTime -> {
                }
            }
            continue
        }

        for ((v, time) in graph[u]) {
            if (v == u) continue

            val nextTime = uTime + time.toDouble()
            if (nextTime == d[v]) {
                dp[v] = (dp[v] + dp[u] % mod) % mod
            }

            if (nextTime < d[v]) {
                dp[v] = dp[u]
                d[v] = nextTime
                queue.add(v to nextTime)
            }
        }
    }
    return (dp[n - 1] % mod).toInt()
}

fun parseEdgeArray(input: String): Array<IntArray> {
    return input
        .removePrefix("[")
        .removeSuffix("]")
        .split("],[")

        // xử lý từng đoạn "a,b,c"
        .map { triplet ->
            triplet
                .replace("[", "")
                .replace("]", "")
                .split(",")
                .map { it.trim().toInt() }
                .toIntArray()
        }
        .toTypedArray()
}

fun maxProbability(n: Int, edges: Array<IntArray>, succProb: DoubleArray, start_node: Int, end_node: Int): Double {
    val graph = Array(n) { mutableListOf<Pair<Int, Double>>() }

    for (i in edges.indices) {
        val (u, v) = edges[i]
        graph[u].add(v to succProb[i])
        graph[v].add(u to succProb[i])
    }

    val d = DoubleArray(n) { 0.0 }
    d[start_node] = 1.0
    val queue = PriorityQueue<Pair<Int, Double>>(compareBy({ it.second }, { -it.first }))
    queue.add(start_node to 1.0)
    while (queue.isNotEmpty()) {
        val (u, uProbability) = queue.poll()
        if (uProbability != d[u]) continue
        //     println("Visit $u ${d[u]}")
        if (u == end_node) {
            // return d[u]
            continue
        }

        for ((v, probability) in graph[u]) {
            if (v == u) continue

            val nextProbability = uProbability * probability
            if (nextProbability > d[v]) {

                d[v] = nextProbability
                queue.add(v to nextProbability)
            }
        }
    }
    return d[end_node]
}

fun nearestExit(maze: Array<CharArray>, entrance: IntArray): Int {
    val directX = intArrayOf(0, 0, 1, -1)
    val directY = intArrayOf(1, -1, 0, 0)
    val m = maze.size
    val n = maze[0].size
    val d = Array(m) { IntArray(n) { Int.MAX_VALUE } }
    d[entrance[0]][entrance[1]] = 0
    val queue = PriorityQueue<Triple<Int, Int, Int>>(
        compareBy(
            { -maxOf(it.first - m, it.second - n, -it.first, -it.second) },
            { -it.third })
    )
    queue.add(Triple(entrance[0], entrance[1], 0))

    var minStep = Int.MAX_VALUE
    while (queue.isNotEmpty()) {
        val (px, py, steps) = queue.poll()

        val isEntrance = px == entrance[0] && py == entrance[1]
        if (!isEntrance && (px == 0 || px == m - 1 || py == 0 || py == n - 1)) {
            minStep = minOf(minStep, steps)
            continue
        }

        if (steps != d[px][py]) continue

        val nextPositions = (0..3).map {
            Pair(px + directX[it], py + directY[it])
        }
        val nextStepCount = d[px][py] + 1

        for ((x, y) in nextPositions) {
            val isValid = x in 0 until m && y in 0 until n && maze[x][y] == '.' && d[x][y] > m * n
            if (isValid && nextStepCount < d[x][y]) {
                d[x][y] = nextStepCount
                queue.add(Triple(x, y, nextStepCount))
            }
        }
    }

    return if (minStep == Int.MAX_VALUE) -1 else minStep
}

fun shortestBridge(grid: Array<IntArray>): Int {
    val n = grid.size
    val firstColor = 2
    val secondColor = 3
    val firstBorders = mutableSetOf<Pair<Int, Int>>()
    val secondBorders = mutableSetOf<Pair<Int, Int>>()

    fun fill(grid: Array<IntArray>, fromX: Int, fromY: Int, baseColors: Set<Int>, color: Int) {
        if (fromX < 0 || fromX > n || grid[fromX][fromY] !in baseColors) return
        grid[fromX][fromY] = color
        val nextList = listOf(
            fromX to fromY + 1,
            fromX to fromY - 1,
            fromX - 1 to fromY,
            fromX + 1 to fromY
        )

        for ((x, y) in nextList) {
            if (x !in 0 until n || y !in 0 until n) continue
            if (grid[x][y] == 0) {
                if (color == firstColor) {
                    firstBorders.add(fromX to fromY)
                } else {
                    secondBorders.add(fromX to fromY)
                }
            }
            if (grid[x][y] !in baseColors) continue
            fill(grid, x, y, baseColors, color)
        }
    }

    var color = firstColor
    for (i in 0 until n) {
        for (j in 0 until n) {
            if (grid[i][j] == 1) {
                fill(grid, i, j, setOf(1), color)
                color = secondColor
            }
        }
    }
    // println(firstBorders)
    //  println(secondBorders)
    var minDistance = Int.MAX_VALUE
    for ((x1, y1) in firstBorders) {
        for ((x2, y2) in secondBorders) {
            val distance = abs(x1 - x2) + abs(y1 - y2)
            minDistance = minOf(distance, minDistance)
        }
    }
    return minDistance - 1
}

fun minMutation(startGene: String, endGene: String, bank: Array<String>): Int {
    if (bank.isEmpty()) {
        return if (startGene == endGene) 0 else -1
    }
    val bankSet = bank.toSet()
    if (endGene !in bankSet) return -1
    fun diff(first: String, second: String): Int {
        var count = 0
        for (i in 0 until 8) {
            if (first[i] != second[i]) count++
        }
        return count
    }

    val list = (bank + startGene + endGene).distinct()
    val n = list.size

    val start = list.indexOf(startGene)
    val end = list.indexOf(endGene)

    if (start < 0 || end < 0) return -1


    val diffs = Array(n) { IntArray(n) }
    val graph = Array(n) { mutableListOf<Int>() }

    for (i in 0 until n - 1) {
        for (j in i + 1 until n) {
            val diff = diff(list[i], list[j])
            diffs[i][j] = diff
            diffs[j][i] = diff

            if (diff == 1) {
                graph[i].add(j)
                graph[j].add(i)
            }
        }
    }
    val d = IntArray(n) { n }
    d[start] = 0
    val queue = PriorityQueue<Int>(compareBy { diffs[it][end] })
    queue.add(start)

    while (queue.isNotEmpty()) {
        val gene = queue.poll()
        if (diffs[gene][end] == 0) {
            return d[end]
        }

        val nextValue = d[gene] + 1
        for (next in graph[gene]) {
            if (nextValue < d[next]) {
                d[next] = nextValue
                queue.add(next)
            }
        }
    }
    return if (d[end] >= n) -1 else d[end]
}

fun ladderLength(beginWord: String, endWord: String, wordList: List<String>): Int {
    if (wordList.isEmpty()) {
        return if (beginWord == endWord) 1 else 0
    }
    val wordSet = wordList.toSet()
    if (endWord !in wordSet) return 0
    fun diff(s1: String, s2: String): Int {
        val n = minOf(s1.length, s2.length)
        var count = 0
        for (i in 0 until n) {
            if (s1[i] != s2[i]) count++
        }
        count += (s1.length - n) + (s2.length - n)
        return count
    }

    val list = if (beginWord in wordSet) wordList else wordList + beginWord
    val n = list.size

    val start = list.indexOf(beginWord)
    val end = list.indexOf(endWord)

    if (start < 0 || end < 0) return 0


    val diffs = Array(n) { IntArray(n) }
    val graph = Array(n) { mutableListOf<Int>() }

    for (i in 0 until n - 1) {
        for (j in i + 1 until n) {
            val diff = diff(list[i], list[j])
            diffs[i][j] = diff
            diffs[j][i] = diff

            if (diff == 1) {
                graph[i].add(j)
                graph[j].add(i)
            }
        }
    }
    val d = IntArray(n) { 2 * n }
    d[start] = 1
    val queue = PriorityQueue<Int>(compareBy { diffs[it][end] })
    queue.add(start)

    val result = mutableListOf<String>()
    var minCount = 2 * n
    while (queue.isNotEmpty()) {
        val gene = queue.poll()
        if (diffs[gene][end] == 0) {
            minCount = minOf(d[end], minCount)
            continue
        }

        val nextValue = d[gene] + 1
        for (next in graph[gene]) {
            if (nextValue < d[next]) {
                d[next] = nextValue
                queue.add(next)
            }
        }
    }
    return if (minCount > n) 0 else minCount
}

fun findLadders(beginWord: String, endWord: String, wordList: List<String>): List<List<String>> {
    fun diffOne(a: String, b: String): Boolean {
        var diff = 0
        for (i in a.indices) {
            if (a[i] != b[i]) {
                if (++diff > 1) return false
            }
        }
        return diff == 1
    }

    if (wordList.isEmpty()) {
        return if (beginWord == endWord) listOf(listOf(beginWord)) else emptyList()
    }
    if (beginWord == endWord) return listOf(listOf(beginWord))
    val words = (wordList + beginWord).toSet().toList()
    val wordToIndex = words.withIndex().associate { (index, word) -> word to index }
    if (wordToIndex[endWord] == null) return emptyList()

    val n = words.size
    val start = wordToIndex[beginWord]
    val end = wordToIndex[endWord]
    if (start == null || end == null) return emptyList()


    //  val diffs = Array(n) { IntArray(n) }
    val graph = Array(n) { mutableListOf<Int>() }

    for (i in 0 until n - 1) {
        for (j in i + 1 until n) {
            val isDiffOne = diffOne(words[i], words[j])
            if (isDiffOne) {
                graph[i].add(j)
                graph[j].add(i)
            }
        }
    }

    val d = IntArray(n) { 2 * n }
    d[start] = 1
    val queue = PriorityQueue<Int>(compareBy { d[it] })
    queue.add(start)

    val predecessors = Array(n) { mutableListOf<Int>() }
//    val sequences = Array(n) { mutableSetOf<List<String>>() }
//    sequences[start] = mutableSetOf(listOf(beginWord))

    while (queue.isNotEmpty()) {
        //    println(" -> ${queue.map { list[it] }}")
        val gene = queue.poll()
        //  println("${list[gene]}: ${queue.map { list[it] }}")
        if (gene == end) {
            continue
        }

        val nextValue = d[gene] + 1

        for (next in graph[gene]) {
            if (nextValue > d[next]) continue

            //   val nextSequences = sequences[gene].map { it + list[next] }
            if (nextValue == d[next]) {
                //  sequences[next].addAll(nextSequences)
                predecessors[next].add(gene)
            }
            if (nextValue < d[next]) {
                predecessors[next] = mutableListOf(gene)
                d[next] = nextValue
                queue.add(next)
            }
        }
    }
    val result = mutableListOf<List<String>>()
    fun backtrack(path: MutableList<String>, node: Int) {
        if (node == start) {
            result.add(path.reversed())
            return
        }
        for (prev in predecessors[node]) {
            path.add(words[prev])
            backtrack(path, prev)
            path.removeLast()
        }
    }
    backtrack(mutableListOf(endWord), end)
    return result
}

fun updateMatrix(mat: Array<IntArray>): Array<IntArray> {
    val m = mat.size
    val n = mat[0].size
    //   println(mat.print())
//   println()
    val grid = Array(m) { IntArray(n) { n * m } }
    fun fill(mat: Array<IntArray>, i: Int, j: Int): Int {
        if (i !in 0 until m || j !in 0 until n) return n * m
        if (mat[i][j] == 0) {
            grid[i][j] = 0
            return 0
        }
        if (mat[i][j] != 1) {
            if (i < m - 1) grid[i][j] = minOf(grid[i][j], 1 + grid[i + 1][j])
            if (j < n - 1) grid[i][j] = minOf(grid[i][j], 1 + grid[i][j + 1])
            if (i > 0) grid[i][j] = minOf(grid[i][j], 1 + grid[i - 1][j])
            if (j > 0) grid[i][j] = minOf(grid[i][j], 1 + grid[i][j - 1])
            return grid[i][j]
        }

        val value = 1 + minOf(
            grid[i][j],
            fill(mat, i, j + 1),
            fill(mat, i, j - 1),
            fill(mat, i - 1, j),
            fill(mat, i + 1, j),
        )
        mat[i][j] = 2
        grid[i][j] = value
        return grid[i][j]
    }

    for (i in 0 until m) {
        for (j in 0 until n) {
            if (mat[i][j] == 0) {
                grid[i][j] = 0
                continue
            }
            if (mat[i][j] == 1) {
                fill(mat, i, j)
            }
        }
    }

    for (i in (m - 1) downTo 0) {
        for (j in (n - 1) downTo 0) {
            if (mat[i][j] > 0) {
                if (i < m - 1) grid[i][j] = minOf(grid[i][j], 1 + grid[i + 1][j])
                if (j < n - 1) grid[i][j] = minOf(grid[i][j], 1 + grid[i][j + 1])
                if (i > 0) grid[i][j] = minOf(grid[i][j], 1 + grid[i - 1][j])
                if (j > 0) grid[i][j] = minOf(grid[i][j], 1 + grid[i][j - 1])
            }
        }
    }

    return grid
}

fun shortestPathBinaryMatrix(grid: Array<IntArray>): Int {
    val m = grid.size
    val n = grid.size
    val directX = intArrayOf(0, 0, 1, 1, 1, -1, -1, -1)
    val directY = intArrayOf(1, -1, -1, 0, 1, -1, 0, 1)
    val d = Array(m) { IntArray(n) { Int.MAX_VALUE } }
    if (grid[0][0] != 0) return -1
    d[0][0] = 1
    val queue = PriorityQueue<Pair<Int, Int>>(
        compareBy(
            { d[it.first][it.second] },
            { abs(m - it.first) + abs(n - it.second) },
        )
    )

    queue.add(0 to 0)
    while (queue.isNotEmpty()) {
        val (px, py) = queue.poll()
        if (px == n - 1 && py == n - 1) {
            return d[px][py]
        }

        val nextCells = mutableListOf<Pair<Int, Int>>()
        for (i in directX) {
            for (j in directY) {
                val nextX = px + i
                val nextY = py + j
                if (nextX in 0 until m && nextY in 0 until n && grid[nextX][nextY] == 0) {
                    nextCells.add(nextX to nextY)
                }
            }
        }
        val nextLength = d[px][py] + 1
        for ((x, y) in nextCells) {
            if (nextLength < d[x][y]) {
                d[x][y] = nextLength
                queue.add(x to y)
            }
        }
    }

    return -1
}

fun distanceK(root: TreeNode?, target: TreeNode?, k: Int): List<Int> {
    target ?: return emptyList()
    root ?: return emptyList()
    val parent = mutableMapOf<TreeNode, TreeNode?>()
    val parentQueue = ArrayDeque<TreeNode>()
    parentQueue.addLast(root)
    while (parentQueue.isNotEmpty()) {
        val node = parentQueue.removeFirst()
        if (node == target) {
            continue
        }
        val leftNode = node.left
        val rightNode = node.right
        if (leftNode != null) {
            parent[leftNode] = node
            parentQueue.addLast(leftNode)
        }
        if (rightNode != null) {
            parent[rightNode] = node
            parentQueue.addLast(rightNode)
        }
    }

    val result = mutableListOf<Int>()
    val prev = mutableMapOf<TreeNode, TreeNode?>()
    val queue = ArrayDeque<Pair<TreeNode, Int>>()
    queue.addLast(target to 0)
    while (queue.isNotEmpty()) {
        val (node, count) = queue.removeFirst()
        if (count == k) {
            result.add(node.`val`)
            continue
        }
        if (count > k) continue

        val leftNode = node.left
        val rightNode = node.right
        val parentNode = parent[node]
        val prevNode = prev[node]
        if (leftNode != null && prevNode != leftNode) {
            prev[leftNode] = node
            queue.addLast(leftNode to count + 1)
        }
        if (rightNode != null && prevNode != rightNode) {
            prev[rightNode] = node
            queue.addLast(rightNode to count + 1)
        }
        if (parentNode != null && prevNode != parentNode) {
            prev[parentNode] = node
            queue.addLast(parentNode to count + 1)
        }
    }
    return result
}

fun amountOfTime(root: TreeNode?, start: Int): Int {
    var startNode = root ?: return 0
    val parent = mutableMapOf<TreeNode, TreeNode?>()
    val parentQueue = ArrayDeque<TreeNode>()
    parentQueue.addLast(root)
    while (parentQueue.isNotEmpty()) {
        val node = parentQueue.removeFirst()
        if (node.`val` == start) {
            startNode = node
        }
        val leftNode = node.left
        val rightNode = node.right
        if (leftNode != null) {
            parent[leftNode] = node
            parentQueue.addLast(leftNode)
        }
        if (rightNode != null) {
            parent[rightNode] = node
            parentQueue.addLast(rightNode)
        }
    }

    val prev = mutableMapOf<TreeNode, TreeNode?>()
    val queue = ArrayDeque<Pair<TreeNode, Int>>()
    queue.addLast(startNode to 0)
    var maxTime = 0
    while (queue.isNotEmpty()) {
        val (node, time) = queue.removeFirst()
        maxTime = maxOf(maxTime, time)

        val leftNode = node.left
        val rightNode = node.right
        val parentNode = parent[node]
        val prevNode = prev[node]
        if (leftNode != null && prevNode != leftNode) {
            prev[leftNode] = node
            queue.addLast(leftNode to time + 1)
        }
        if (rightNode != null && prevNode != rightNode) {
            prev[rightNode] = node
            queue.addLast(rightNode to time + 1)
        }
        if (parentNode != null && prevNode != parentNode) {
            prev[parentNode] = node
            queue.addLast(parentNode to time + 1)
        }
    }
    return maxTime
}

fun numEnclaves(grid: Array<IntArray>): Int {
    val m = grid.size
    val n = grid[0].size

    fun fill(x: Int, y: Int) {
        if (x !in 0 until m || y !in 0 until n) return
        if (grid[x][y] != 1) return
        grid[x][y] = 0
        if (x < m - 1) fill(x + 1, y)
        if (x > 0) fill(x - 1, y)
        if (y < n - 1) fill(x, y + 1)
        if (y > 0) fill(x, y - 1)
    }

    for (i in 0 until m) {
        if (grid[i][0] == 1) fill(i, 0)
        if (grid[i][n - 1] == 1) fill(i, n - 1)
    }
    for (j in 0 until n) {
        if (grid[0][j] == 1) fill(0, j)
        if (grid[m - 1][j] == 1) fill(m - 1, j)
    }
    var count = 0
    for (i in 0 until m) {
        for (j in 0 until n) {
            if (grid[i][j] == 1) {
                count++
            }
        }
    }
    return count
}

fun closedIsland(grid: Array<IntArray>): Int {
    val m = grid.size
    val n = grid[0].size

    fun fill(x: Int, y: Int) {
        if (x !in 0 until m || y !in 0 until n) return
        if (grid[x][y] != 0) return
        grid[x][y] = 1
        if (x < m - 1) fill(x + 1, y)
        if (x > 0) fill(x - 1, y)
        if (y < n - 1) fill(x, y + 1)
        if (y > 0) fill(x, y - 1)
    }

    for (i in 0 until m) {
        if (grid[i][0] == 0) fill(i, 0)
        if (grid[i][n - 1] == 0) fill(i, n - 1)
    }
    for (j in 0 until n) {
        if (grid[0][j] == 0) fill(0, j)
        if (grid[m - 1][j] == 0) fill(m - 1, j)
    }
    var count = 0
    for (i in 0 until m) {
        for (j in 0 until n) {
            if (grid[i][j] == 0) {
                count++
                fill(i, j)
            }
        }
    }
    return count
}

fun pacificAtlantic(heights: Array<IntArray>): List<List<Int>> {
    val m = heights.size
    val n = heights[0].size
    val grid = Array(m) { x ->
        IntArray(n) { y ->
            var value = 0
            if (x == 0 || y == 0) value = value or 1
            if (x == m - 1 || y == n - 1) value = value or 2
            value
        }
    }

    var visited = Array(m) { BooleanArray(n) }

    fun fill(x: Int, y: Int, value: Int) {
        if (x !in 0 until m || y !in 0 until n) return
        grid[x][y] = grid[x][y] or value
        if (visited[x][y]) return
        visited[x][y] = true

        val currentHeight = heights[x][y]
        if (x < m - 1 && heights[x + 1][y] >= currentHeight) {
            fill(x + 1, y, grid[x][y])
        }
        if (x > 0 && heights[x - 1][y] >= currentHeight) {
            fill(x - 1, y, grid[x][y])
        }
        if (y > 0 && heights[x][y - 1] >= currentHeight) {
            fill(x, y - 1, grid[x][y])
        }
        if (y < n - 1 && heights[x][y + 1] >= currentHeight) {
            fill(x, y + 1, grid[x][y])
        }
    }

    for (i in 0 until m) {
        if (!visited[i][0]) fill(i, 0, grid[i][0])
    }
    visited = Array(m) { BooleanArray(n) }
    for (i in 0 until m) {
        if (!visited[i][n - 1]) fill(i, n - 1, grid[i][n - 1])
    }
    visited = Array(m) { BooleanArray(n) }

    for (j in 0 until n) {
        if (!visited[m - 1][j]) fill(m - 1, j, grid[m - 1][j])
    }
    visited = Array(m) { BooleanArray(n) }

    for (j in 0 until n) {
        if (!visited[0][j]) fill(0, j, grid[0][j])
    }

    val result = mutableListOf<List<Int>>()
    for (i in 0 until m) {
        for (j in 0 until n) {
            if (grid[i][j] == 3) {
                result.add(listOf(i, j))
            }
        }
    }
    return result
}

fun findJudge(n: Int, trust: Array<IntArray>): Int {
    if (n == 1 && trust.isEmpty()) return -1
    val score = IntArray(n + 1)

    for ((a, b) in trust) {
        score[a]--
        score[b]++
    }
    return score.indexOf(n - 1)
}

fun constructProductMatrix(grid: Array<IntArray>): Array<IntArray> {
    val m = grid.size
    val n = grid[0].size
    val size = m * n
    val mod = 12345

    val prefix = LongArray(size)
    val suffix = LongArray(size)

    for (index in 0 until size) {
        val i = index / n
        val j = index % n
        val value = grid[i][j] % mod
        prefix[index] = if (index == 0) value.toLong() else (prefix[index - 1] * value) % mod
    }

    for (index in size - 1 downTo 0) {
        val i = index / n
        val j = index % n
        val value = grid[i][j] % mod
        suffix[index] = if (index == size - 1) value.toLong() else (suffix[index + 1] * value) % mod
    }

    val result = Array(m) { IntArray(n) }
    for (index in 0 until size) {
        val i = index / n
        val j = index % n
        val before = if (index > 0) prefix[index - 1] else 1L
        val after = if (index < size - 1) suffix[index + 1] else 1L
        result[i][j] = ((before * after) % mod).toInt()
    }

    return result
}

fun findSmallestSetOfVertices(n: Int, edges: List<List<Int>>): List<Int> {
    val graph = Array(n) { mutableListOf<Int>() }

    for ((v, u) in edges) {
        graph[u].add(v)
    }

    val visited = BooleanArray(n)
    val result = mutableListOf<Int>()
    fun dfs(u: Int) {
        if (visited[u]) return
        visited[u] = true

        val nextNodes = graph[u].filterNot { visited[it] }
        if (nextNodes.isEmpty()) {
            result.add(u)
        }

        for (v in nextNodes) {
            dfs(v)
        }
    }

    for (i in 0 until n) {
        if (graph[i].isEmpty()) {
            dfs(i)
        }
    }
    return result
}

fun maxArea(heights: IntArray): Int {
    val n = heights.size
//
//    val stack = Stack<Int>()
//    val rights = IntArray(n)
//    for (i in 0 until n) {
//        while (stack.isNotEmpty() && heights[i] > heights[stack.peek()]) {
//            val top = stack.pop()
//            rights[top] = i
//        }
//        stack.push(i)
//    }
//    while (stack.isNotEmpty()) {
//        rights[stack.pop()] = n
//    }
//    stack.clear()
//    val lefts = IntArray(n)
//    for (i in (n - 1) downTo 0) {
//        while (stack.isNotEmpty() && heights[i] > heights[stack.peek()]) {
//            val top = stack.pop()
//            lefts[top] = i
//        }
//        stack.push(i)
//    }
//    while (stack.isNotEmpty()) {
//        lefts[stack.pop()] = -1
//    }

    //   println((0 until n).toList())
    //   println(heights.toList())
    //   println(rights.toList())
    //  println(lefts.toList())

    var l = 0
    var r = n - 1
    var maxArea = 0
    while (l < r && l in 0 until n && r in 0 until n) {
        val area = minOf(heights[l], heights[r]) * (r - l)
        maxArea = maxOf(maxArea, area)

        if (heights[l] > heights[r]) {
            r--
        } else {
            l++
        }

        //  println("$l $r $area")
//        val nLeft = rights[l]
//        val lArea = if (nLeft < r && nLeft in 0 until n) {
//            (r - nLeft) * minOf(heights[nLeft], heights[r])
//        } else -1
//
//        var nRight = r
//        val rArea = if (nRight > l && nRight in 0 until n) {
//            (nRight - l) * minOf(heights[nRight], heights[l])
//        } else -1
//
//        when {
//            lArea <= 0 && rArea <= 0 -> break
//            lArea <= 0 -> r = nRight
//            rArea <= 0 -> l = nLeft
//            lArea >= rArea -> l = nLeft
//            else -> r = nRight
//        }
    }
    return maxArea

}

fun main() {
    println(
        maxArea(
            intArrayOf(7, 10, 6, 2, 5, 4, 8, 3, 7)
        )
    )
}