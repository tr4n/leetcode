package remote

import java.util.*

class PrefixSum2D(private val matrix: Array<IntArray>) {

    private val m = matrix.size
    private val n = matrix[0].size
    private val prefix = Array(m) { IntArray(n) }

    init {
        buildPrefixSum()
    }

    private fun buildPrefixSum() {
        for (i in 0 until m) {
            for (j in 0 until n) {
                val top = if (i > 0) prefix[i - 1][j] else 0
                val left = if (j > 0) prefix[i][j - 1] else 0
                val corner = if (i > 0 && j > 0) prefix[i - 1][j - 1] else 0
                prefix[i][j] = matrix[i][j] + top + left - corner
            }
        }
    }

    fun query(r1: Int, c1: Int, r2: Int, c2: Int): Int {
        val total = prefix[r2][c2]
        val top = if (r1 > 0) prefix[r1 - 1][c2] else 0
        val left = if (c1 > 0) prefix[r2][c1 - 1] else 0
        val corner = if (r1 > 0 && c1 > 0) prefix[r1 - 1][c1 - 1] else 0
        return total - top - left + corner
    }
}

fun numberOfSubmatrices(grid: Array<CharArray>): Int {
    val m = grid.size
    val n = grid[0].size
    val tableY = Array(m) { i ->
        IntArray(n) { j ->
            when (grid[i][j]) {
                'Y' -> 1
                else -> 0
            }
        }
    }
    val tableX = Array(m) { i ->
        IntArray(n) { j ->
            when (grid[i][j]) {
                'X' -> 1
                else -> 0
            }
        }
    }

    val xCounter = PrefixSum2D(tableX)
    val yCounter = PrefixSum2D(tableY)


    var cnt = 0
    for (i in 0 until m) {
        for (j in 0 until n) {
            val x = xCounter.query(0, 0, i, j)
            val y = yCounter.query(0, 0, i, j)
            if (x > 0 && x == y) cnt++
        }
    }
    return cnt
}

class PrefixSum2DLong(private val matrix: Array<IntArray>) {

    private val m = matrix.size
    private val n = matrix[0].size
    private val prefix = Array(m) { LongArray(n) }

    init {
        buildPrefixSum()
    }

    private fun buildPrefixSum() {
        for (i in 0 until m) {
            for (j in 0 until n) {
                val top = if (i > 0) prefix[i - 1][j] else 0
                val left = if (j > 0) prefix[i][j - 1] else 0
                val corner = if (i > 0 && j > 0) prefix[i - 1][j - 1] else 0
                prefix[i][j] = matrix[i][j].toLong() + top + left - corner
            }
        }
    }

    fun query(r1: Int, c1: Int, r2: Int, c2: Int): Long {
        val total = prefix[r2][c2]
        val top = if (r1 > 0) prefix[r1 - 1][c2] else 0
        val left = if (c1 > 0) prefix[r2][c1 - 1] else 0
        val corner = if (r1 > 0 && c1 > 0) prefix[r1 - 1][c1 - 1] else 0
        return total - top - left + corner
    }
}

fun numSubmat(mat: Array<IntArray>): Int {
    val prefixSum = PrefixSum2D(mat)
    val m = mat.size
    val n = mat[0].size

    val stack = Stack<Int>()
    var cnt = 0
    val leftIndex = Array(m) { i -> IntArray(n) { -1 } }
    for (i in 0 until m) {
        stack.clear()
        for (j in (n - 1) downTo 0) {
            while (stack.isNotEmpty() && mat[i][j] < mat[i][stack.peek()]) {
                val end = stack.pop()
                leftIndex[i][end] = j + 1
            }
            if (mat[i][j] == 1) stack.push(j)
        }
        while (stack.isNotEmpty()) {
            val end = stack.pop()
            leftIndex[i][end] = 0
        }
    }

    val topIndex = Array(m) { IntArray(n) { -1 } }
    for (j in 0 until n) {
        stack.clear()
        for (i in (m - 1) downTo 0) {
            while (stack.isNotEmpty() && mat[i][j] < mat[stack.peek()][j]) {
                val end = stack.pop()
                topIndex[end][j] = i + 1
            }
            if (mat[i][j] == 1) stack.push(i)
        }
        while (stack.isNotEmpty()) {
            val end = stack.pop()
            topIndex[end][j] = 0
        }
    }

    for (i in 0 until m) {
        for (j in 0 until n) {
            if (mat[i][j] == 0) continue
            for (x in topIndex[i][j]..i) {
                for (y in leftIndex[i][j]..j) {
                    val width = i - x + 1
                    val height = j - y + 1
                    val sum = prefixSum.query(x, y, i, j)
                    if (sum == width * height) cnt++
                }
            }
        }
    }

//    println(mat.print())
//    println("")
//    println(leftIndex.print())
//    println()
//    println(topIndex.print())

    return cnt
}


fun countSquares(matrix: Array<IntArray>): Int {
    val prefixSum = PrefixSum2D(matrix)
    val m = matrix.size
    val n = matrix[0].size

    var cnt = 0
    for (i in 0 until m) {
        for (j in 0 until n) {
            val maxSize = 1 + minOf(i, j)
            for (size in 1..maxSize) {
                val x = i - size + 1
                val y = j - size + 1
                val sum = prefixSum.query(x, y, i, j)
                if (sum == size * size) cnt++
            }
        }
    }
    return cnt
}

fun maxSumSubmatrix(matrix: Array<IntArray>, k: Int): Int {
    val m = matrix.size
    val n = matrix[0].size
    val prefixSum2D = PrefixSum2D(matrix)
    //  println(matrix.print())
    var minDiff = Int.MAX_VALUE
    var closetSum = 0
    for (row in 0 until m) {
        for (i in 0..row) {

            val tree = TreeSet<Int>()
            tree.add(0)
            var sum = 0
            for (col in 0 until n) {
                sum += prefixSum2D.query(i, col, row, col)
                //    println("row($i $row), col ($col) $sum")
                if (sum == k) return k
                var diff = k - sum
                if (diff in 0..<minDiff) {
                    minDiff = diff
                    closetSum = sum
                }
                // sum - left <= k -> left >= sum - k
                val leftSum = tree.ceiling(sum - k)
                if (leftSum == null) {
                    tree.add(sum)
                    continue
                }
                val rectSum = sum - leftSum
                if (rectSum == k) return k
                diff = k - rectSum
                if (diff in 0..<minDiff) {
                    minDiff = diff
                    closetSum = rectSum
                }
                tree.add(sum)
            }
            //           println("row($i $row), $tree")
        }
    }
    return closetSum
}

fun matrixBlockSum(mat: Array<IntArray>, k: Int): Array<IntArray> {
    val m = mat.size
    val n = mat[0].size
    val prefixSum2D = PrefixSum2D(mat)
    return Array(m) { i ->
        IntArray(n) { j ->
            prefixSum2D.query(
                (i - k).coerceIn(0, m - 1),
                (j - k).coerceIn(0, n - 1),
                (i + k).coerceIn(0, m - 1),
                (j + k).coerceIn(0, n - 1)
            )
        }
    }
}

fun maxSideLength(mat: Array<IntArray>, threshold: Int): Int {
    val prefixSum = PrefixSum2D(mat)
    val m = mat.size
    val n = mat[0].size

    var sizeLength = 0
    for (i in 0 until m) {
        for (j in 0 until n) {
            val maxSize = 1 + minOf(i, j)
            for (size in maxSize downTo 1) {
                val x = i - size + 1
                val y = j - size + 1
                val sum = prefixSum.query(x, y, i, j)
                if (sum <= threshold) {
                    sizeLength = maxOf(sizeLength, size)
                }
            }
        }
    }
    return sizeLength
}

fun numSubmatrixSumTarget(matrix: Array<IntArray>, target: Int): Int {
    val m = matrix.size
    val n = matrix[0].size
    val prefixSum2D = PrefixSum2D(matrix)
    //  println(matrix.print())
    var cnt = 0
    for (row in 0 until m) {
        for (i in 0..row) {

            val tree = mutableMapOf<Int, Int>()
            var sum = 0
            for (col in 0 until n) {
                sum += prefixSum2D.query(i, col, row, col)
                //    println("row($i $row), col ($col) $sum")
                if (sum == target) {
                    cnt++
                }

                // sum - left == k -> left == sum - k
                val count = tree.getOrDefault(sum - target, 0)
                cnt += count
                tree[sum] = (tree[sum] ?: 0) + 1
            }
            //           println("row($i $row), $tree")
        }
    }
    return cnt
}