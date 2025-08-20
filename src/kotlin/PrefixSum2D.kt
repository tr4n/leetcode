package org.example

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

