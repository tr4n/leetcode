package remote

class NumMatrix(matrix: Array<IntArray>) {
    private val rows = matrix.size
    private val cols = matrix[0].size
    private val prefixSum = Array(rows) { IntArray(cols) }

    init {
        for (r in 0 until rows) {
            for (c in 0 until cols) {
                val above = if (r > 0) prefixSum[r - 1][c] else 0
                val left = if (c > 0) prefixSum[r][c - 1] else 0
                val diag = if (r > 0 && c > 0) prefixSum[r - 1][c - 1] else 0
                prefixSum[r][c] = matrix[r][c] + above + left - diag
            }
        }
    }

    fun sumRegion(row1: Int, col1: Int, row2: Int, col2: Int): Int {
        val minX = minOf(row1, row2)
        val minY = minOf(col1, col2)

        val maxX = maxOf(row1, row2)
        val maxY = maxOf(col1, col2)

        var result = prefixSum[maxX][maxY]
        if (minX > 0) result -= prefixSum[minX - 1][maxY]
        if (minY > 0) result -= prefixSum[maxX][minY - 1]
        if (minX > 0 && minY > 0) result += prefixSum[minX - 1][minY - 1]

        return result
    }
}
