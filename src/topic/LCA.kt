package topic

import kotlin.math.log2

class LCA(
    private val n: Int,
    edges: Array<IntArray>,
    private val root: Int = 0
) {
    private val maxLog = 1 + log2(n.toDouble()).toInt()
    private val graph = Array(n) { mutableListOf<Int>() }
    private val depth = IntArray(n)
    private val up = Array(n) { IntArray(maxLog) }

    init {
        for ((u, v) in edges) {
            graph[u].add(v)
            graph[v].add(u)
        }
        preprocess()
    }

    private fun preprocess() {
        dfs(root, -1, 0)
        for (j in 1 until maxLog) {
            for (i in 0 until n) {
                val k = up[i][j - 1]
                up[i][j] = up[k][j - 1]
            }
        }
    }

    private fun dfs(u: Int, p: Int, d: Int) {
        depth[u] = d
        up[u][0] = p
        for (v in graph[u]) {
            if (v == p) continue
            dfs(v, u, d + 1)
        }
    }

    fun query(nodeU: Int, nodeV: Int): Int {
        var u = nodeU
        var v = nodeV

        if (depth[u] < depth[v]) {
            val temp = u
            u = v
            v = temp
        }

        for (j in (maxLog - 1) downTo 0) {
            if (depth[u] - (1 shl j) >= depth[v]) {
                u = up[u][j]
            }
        }

        if (u == v) return u

        for (j in (maxLog - 1) downTo 0) {
            if (up[u][j] == up[v][j]) continue
            u = up[u][j]
            v = up[v][j]
        }
        return up[u][0]
    }

}

class TreeAncestor(private val n: Int, private val parent: IntArray) {
    private val maxLog = 1 + log2(n.toDouble()).toInt()
    private val graph = Array(n) { mutableListOf<Int>() }
    private val depth = IntArray(n)
    private val up = Array(n) { IntArray(maxLog) { -1} }

    init {
        for (u in 0 until n) {
            val p = parent[u]
            if (p != -1) graph[p].add(u)
        }
        preprocess()
    }

    private fun preprocess() {
        dfs(0, -1, 0)
        for (j in 1 until maxLog) {
            for (i in 0 until n) {
                val k = up[i][j - 1]
                if (k == -1) continue
                up[i][j] = up[k][j - 1]
            }
        }
    }

    private fun dfs(u: Int, p: Int, d: Int) {
        depth[u] = d
        up[u][0] = p
        for (v in graph[u]) {
            if (v == p) continue
            dfs(v, u, d + 1)
        }
    }

    fun query(nodeU: Int, nodeV: Int): Int {
        var u = nodeU
        var v = nodeV

        if (depth[u] < depth[v]) {
            val temp = u
            u = v
            v = temp
        }

        for (j in (maxLog - 1) downTo 0) {
            if (depth[u] - (1 shl j) >= depth[v]) {
                u = up[u][j]
            }
        }

        if (u == v) return u

        for (j in (maxLog - 1) downTo 0) {
            if (up[u][j] == up[v][j]) continue
            u = up[u][j]
            v = up[v][j]
        }
        return up[u][0]
    }

    fun getKthAncestor(node: Int, k0: Int): Int {
        var u = node
        var k = k0
        var bit = 0
        while (k > 0 && u != -1) {
            if ((k and 1) == 1) u = up[u][bit]
            k = k shr 1
            bit++
        }
        return u
    }
}

fun lowestCommonAncestor(root: TreeNode?, p: TreeNode?, q: TreeNode?): TreeNode? {
    var ans: TreeNode? = null
    fun dfs(node: TreeNode?): Int {
        node ?: return 0
        var sum = 0
        if (node == p) sum = sum or 1
        if (node == q) sum = sum or 2
        if (sum == 3) return sum
        val left = dfs(node.left)
        val right = dfs(node.right)
        sum = sum or left or right
        if (sum == 3 && ans == null) ans = node
        return sum
    }
    dfs(root)

    return ans ?: p
}

fun main() {
    // Ví dụ đầu vào
    val n = 9
    val edges = arrayOf(
        intArrayOf(0, 1), intArrayOf(0, 2),
        intArrayOf(1, 3), intArrayOf(1, 4),
        intArrayOf(2, 5),
        intArrayOf(4, 6), intArrayOf(4, 7),
        intArrayOf(5, 8)
    )

    // Khởi tạo đối tượng LCA, quá trình tiền xử lý sẽ tự động chạy
    val lcaSolver = LCA(n, edges)

    // Thực hiện các truy vấn
    val u1 = 6
    val v1 = 8
    println("LCA of $u1 and $v1 is: ${lcaSolver.query(u1, v1)}") // Kết quả: 0

    val u2 = 7
    val v2 = 3
    println("LCA of $u2 and $v2 is: ${lcaSolver.query(u2, v2)}") // Kết quả: 1
}