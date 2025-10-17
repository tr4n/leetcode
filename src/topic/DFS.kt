package topic

fun sumOfDistancesInTree(n: Int, edges: Array<IntArray>): IntArray {
    val graph = Array(n) { mutableListOf<Int>() }
    for ((u, v) in edges) {
        graph[u].add(v)
        graph[v].add(u)
    }

    val subTreeSize = IntArray(n) { 1 }
    val sumDistDown = IntArray(n)

    fun dfs(u: Int, parent: Int) {
        for (v in graph[u]) {
            if (v == parent) continue
            dfs(v, u)
            subTreeSize[u] += subTreeSize[v]
            sumDistDown[u] += sumDistDown[v] + subTreeSize[v]
        }
    }

    val result = IntArray(n)
    fun calc(u: Int, parent: Int) {
        for (v in graph[u]) {
            if (v == parent) continue
            result[v] = result[u] - 2 * subTreeSize[v] + n
            calc(v, u)
        }
    }
    dfs(0, -1)
    result[0] = sumDistDown[0]
    calc(0, -1)
    return result
}

fun componentValue(nums: IntArray, edges: Array<IntArray>): Int {
    val n = nums.size
    val total = nums.sum()

    val graph = Array(n) { mutableListOf<Int>() }
    for ((u, v) in edges) {
        graph[u].add(v)
        graph[v].add(u)
    }

    var seen = BooleanArray(n)
    var cnt = 0
    fun dfs(u: Int, target: Int): Int {
        seen[u] = true
        var sum = nums[u]
        for (v in graph[u]) {
            if (seen[v]) continue
            val subSum = dfs(v, target)
            if (subSum < 0) return -1
            sum += subSum
        }
        if (sum == target) {
            cnt++
            return 0
        }
        if (sum > target) {
            cnt = -1
            return -1
        }
        return sum
    }

    for (target in 1..total) {
        if (total % target != 0) continue
        seen = BooleanArray(n)
        cnt = 0
        dfs(0, target)
        if (cnt == total / target) {
            return cnt - 1
        }
    }
    return 0
}

fun maxKDivisibleComponents(n: Int, edges: Array<IntArray>, values: IntArray, k: Int): Int {
    val graph = Array(n) { mutableListOf<Int>() }
    for ((u, v) in edges) {
        graph[u].add(v)
        graph[v].add(u)
    }

    var seen = BooleanArray(n)
    var cnt = 0
    fun dfs(u: Int): Long {
        seen[u] = true
        var sum = values[u].toLong()
        for (v in graph[u]) {
            if (seen[v]) continue
            sum += dfs(v)
        }
        if (sum % k == 0L) {
            cnt++
            return 0
        }
        return sum
    }

    seen = BooleanArray(n)
    cnt = 0
    dfs(0)
    return cnt
}

fun maximumScore(scores: IntArray, edges: Array<IntArray>): Int {
    val n = scores.size
    val graph = Array(n) { mutableListOf<Int>() }
    for ((u, v) in edges) {
        graph[u].add(v)
        graph[v].add(u)
    }

    for (u in 0 until n) {
        graph[u].sortByDescending { scores[it] }
    }
    var ans = -1
    for ((u, v) in edges) {
        for (i in 0 until minOf(graph[u].size, 3)) {
            for (j in 0 until minOf(graph[v].size, 3)) {
                val p = graph[u][i]
                val q = graph[v][j]
                if (p == q || p == v || q == u) continue
                val score = scores[u] + scores[v] + scores[p] + scores[q]
                ans = maxOf(ans, score)
                break
            }
        }
    }

    return ans
}

fun countSubgraphsForEachDiameter(n: Int, edges: Array<IntArray>): IntArray {
    val limit = 1 shl n
    val graph = Array(n) { mutableListOf<Int>() }
    for ((u, v) in edges) {
        graph[u - 1].add(v - 1)
        graph[v - 1].add(u - 1)
    }

    fun bfs(start: Int, mask: Int): Pair<Int, Int> {
        var maxDist = 0
        var farthest = start

        val dist = IntArray(n) { -1 }
        dist[start] = 0
        val queue = ArrayDeque<Int>()
        queue.add(start)

        while (queue.isNotEmpty()) {
            val u = queue.removeFirst()
            if (maxDist < dist[u]) {
                maxDist = dist[u]
                farthest = u
            }

            for (v in graph[u]) {
                if ((mask shr v) and 1 == 0) continue
                if (dist[v] >= 0) continue

                dist[v] = dist[u] + 1
                queue.add(v)
            }
        }
        return farthest to maxDist
    }

    fun distance(mask: Int): Int {
        var u = 0
        while ((mask shr u) and 1 == 0) u++
        val v = bfs(u, mask).first
        return bfs(v, mask).second
    }

    fun dfs(u: Int, mask: Int): Int {
        var cnt = 1
        val newMask = mask and (1 shl u).inv()

        for (v in graph[u]) {
            if ((newMask shr v) and 1 == 0) continue

            cnt += dfs(v, newMask)
        }
        return cnt
    }

    fun isConnected(mask: Int): Boolean {
        val vertexCount = mask.countOneBits()
        var start = 0
        while ((mask shr start) and 1 == 0) start++
        return vertexCount == dfs(start, mask)
    }

    val result = IntArray(n - 1)

    for (mask in 1 until limit) {
        if (mask.countOneBits() <= 1) continue
        if (!isConnected(mask)) continue
        val dist = distance(mask)
        if (dist > 0) result[dist - 1]++
    }
    return result
}

fun queensAttacktheKing(queens: Array<IntArray>, king: IntArray): List<List<Int>> {
    val grid = Array(8) { BooleanArray(8) }
    for (queen in queens) {
        grid[queen[0]][queen[1]] = true
    }

    val dirX = intArrayOf(-1, -1, -1, 0, 0, 1, 1, 1)
    val dirY = intArrayOf(-1, 0, 1, -1, 1, -1, 0, 1)

    val r = king[0]
    val c = king[1]
    val result = mutableListOf<List<Int>>()
    for (i in 0 until 8) {
        var x = r
        var y = c
        while (x in 0 until 8 && y in 0 until 8) {
            x += dirX[i]
            y += dirY[i]
            if (grid[x][y]) {
                result.add(listOf(x, y))
                break
            }
        }
    }
    return result
}

fun minMovesToCaptureTheQueen(
    rookX: Int, rookY: Int,
    bishopX: Int, bishopY: Int,
    queenX: Int, queenY: Int
): Int {
    val sameColumn = (rookX == queenX)
    val sameRow = (rookY == queenY)

    val bishopBlocksColumn = (bishopX == rookX &&
            bishopY in minOf(rookY, queenY)..maxOf(rookY, queenY))
    val bishopBlocksRow = (bishopY == rookY &&
            bishopX in minOf(rookX, queenX)..maxOf(rookX, queenX))

    val rookCanCapture = (sameColumn && !bishopBlocksColumn) ||
            (sameRow && !bishopBlocksRow)
    if (rookCanCapture) return 1

    val sameDiagonal1 = (bishopX - bishopY == queenX - queenY)
    val sameDiagonal2 = (bishopX + bishopY == queenX + queenY)

    val rookBlocksDiagonal1 = (rookX - rookY == bishopX - bishopY &&
            rookX - rookY == queenX - queenY &&
            rookX in minOf(bishopX, queenX)..maxOf(bishopX, queenX) &&
            rookY in minOf(bishopY, queenY)..maxOf(bishopY, queenY))

    val rookBlocksDiagonal2 = (rookX + rookY == bishopX + bishopY &&
            rookX + rookY == queenX + queenY &&
            rookX in minOf(bishopX, queenX)..maxOf(bishopX, queenX) &&
            rookY in minOf(bishopY, queenY)..maxOf(bishopY, queenY))

    val bishopCanCapture = (sameDiagonal1 || sameDiagonal2) &&
            !(rookBlocksDiagonal1 || rookBlocksDiagonal2)
    if (bishopCanCapture) return 1

    return 2
}

fun buildTreeWithInorder(preorder: IntArray, inorder: IntArray): TreeNode? {
    val n = preorder.size
    val inorderMap = inorder.withIndex().associate { it.value to it.index }
    var index = 0

    fun build(start: Int, end: Int): TreeNode? {
        if (index >= n) return null
        val value = preorder[index++]
        val root = TreeNode(value)

        val mid = inorderMap[value] ?: 0
        if (mid > start) root.left = build(start, mid - 1)
        if (mid < end) root.right = build(mid + 1, end)
        return root
    }

    return build(0, n - 1)
}

fun buildTree(inorder: IntArray, postorder: IntArray): TreeNode? {
    val n = postorder.size
    val inorderMap = inorder.withIndex().associate { it.value to it.index }
    var index = n - 1

    fun build(start: Int, end: Int): TreeNode? {
        if (index < 0) return null
        val value = postorder[index--]
        val root = TreeNode(value)

        val mid = inorderMap[value] ?: return null
        if (mid < end) root.right = build(mid + 1, end)
        if (mid > start) root.left = build(start, mid - 1)
        return root
    }

    return build(0, n - 1)
}

fun reachingPoints(sx: Int, sy: Int, tx: Int, ty: Int): Boolean {

    fun dfs(x: Int, y: Int): Boolean {
        if (x == sx && y == sy) return true
        if (x < sx || y < sy) return false

        return if (x > y) {
            if (y == sy) (x - sx) % y == 0
            else dfs(x % y, y)
        } else {
            if (x == sx) (y - sy) % x == 0
            else dfs(x, y % x)
        }
    }

    return dfs(tx, ty)
}