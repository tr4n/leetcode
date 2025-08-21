package remote

import local.to2DIntArray
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
            if(x > 0 && x == y) cnt ++
        }
    }
    return cnt
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

fun main() {
    println(
        numSubmat(
            "[[0,1,1,0],[0,1,1,1],[1,1,1,0]]".to2DIntArray()
        )
    )
}

