package remote

import local.to2DIntArray
import java.util.*
import kotlin.math.max
import kotlin.math.min

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
//fun minSubmatrixSum(minX: Int, minY: Int, maxX: Int, maxY: Int): Int {
//    var minSum = Int.MAX_VALUE
//    for (row in minX until maxX) {
//        val colSums = IntArray(maxY - minY)
//        for (i in row until maxX) {
//            for (col in minY until maxY) {
//                colSums[col - minY] += prefixSum2D.query(i, col, i, col)
//            }
//            var current = 0
//            for (v in colSums) {
//                current = minOf(v, current + v)
//                minSum = minOf(minSum, current)
//            }
//        }
//    }
//    return minSum
//}

fun minimumSum(grid: Array<IntArray>): Int {
    val rows = grid.size
    val cols = if (grid.isNotEmpty()) grid[0].size else 0
    if (rows == 0 || cols == 0) return 0

    val rowBBoxes = Array(rows) { Pair(Int.MAX_VALUE, -1) }
    for (r in 0 until rows) {
        var minC = Int.MAX_VALUE
        var maxC = -1
        for (c in 0 until cols) {
            if (grid[r][c] == 1) {
                minC = min(minC, c)
                maxC = max(maxC, c)
            }
        }
        rowBBoxes[r] = Pair(minC, maxC)
    }

    val colBBoxes = Array(cols) { Pair(Int.MAX_VALUE, -1) }
    for (c in 0 until cols) {
        var minR = Int.MAX_VALUE
        var maxR = -1
        for (r in 0 until rows) {
            if (grid[r][c] == 1) {
                minR = min(minR, r)
                maxR = max(maxR, r)
            }
        }
        colBBoxes[c] = Pair(minR, maxR)
    }

    fun getHorizontalSliceArea(r1: Int, r2: Int): Int {
        var minR = Int.MAX_VALUE
        var maxR = -1
        var minC = Int.MAX_VALUE
        var maxC = -1
        var hasOne = false

        for (r in r1..r2) {
            val rowBound = rowBBoxes[r]
            if (rowBound.first <= rowBound.second) { // Has one
                hasOne = true
                minR = min(minR, r)
                maxR = max(maxR, r)
                minC = min(minC, rowBound.first)
                maxC = max(maxC, rowBound.second)
            }
        }
        return if (hasOne) (maxR - minR + 1) * (maxC - minC + 1) else 0
    }

    fun getVerticalSliceArea(c1: Int, c2: Int): Int {
        var minR = Int.MAX_VALUE
        var maxR = -1
        var minC = Int.MAX_VALUE
        var maxC = -1
        var hasOne = false

        for (c in c1..c2) {
            val colBound = colBBoxes[c]
            if (colBound.first <= colBound.second) {
                hasOne = true
                minR = min(minR, colBound.first)
                maxR = max(maxR, colBound.second)
                minC = min(minC, c)
                maxC = max(maxC, c)
            }
        }
        return if (hasOne) (maxR - minR + 1) * (maxC - minC + 1) else 0
    }

    fun getBoundingBoxArea(r1: Int, c1: Int, r2: Int, c2: Int): Int {
        var minRow = Int.MAX_VALUE
        var maxRow = -1
        var minCol = Int.MAX_VALUE
        var maxCol = -1
        var hasOne = false
        for (r in r1..r2) {
            for (c in c1..c2) {
                if (grid[r][c] == 1) {
                    hasOne = true
                    minRow = min(minRow, r)
                    maxRow = max(maxRow, r)
                    minCol = min(minCol, c)
                    maxCol = max(maxCol, c)
                }
            }
        }
        return if (hasOne) (maxRow - minRow + 1) * (maxCol - minCol + 1) else 0
    }


    var minTotalArea = Int.MAX_VALUE

    if (rows >= 3) {
        for (r1 in 0 until rows - 2) {
            for (r2 in r1 + 1 until rows - 1) {
                val area1 = getHorizontalSliceArea(0, r1)
                val area2 = getHorizontalSliceArea(r1 + 1, r2)
                val area3 = getHorizontalSliceArea(r2 + 1, rows - 1)
                minTotalArea = min(minTotalArea, area1 + area2 + area3)
            }
        }
    }

    if (cols >= 3) {
        for (c1 in 0 until cols - 2) {
            for (c2 in c1 + 1 until cols - 1) {
                val area1 = getVerticalSliceArea(0, c1)
                val area2 = getVerticalSliceArea(c1 + 1, c2)
                val area3 = getVerticalSliceArea(c2 + 1, cols - 1)
                minTotalArea = min(minTotalArea, area1 + area2 + area3)
            }
        }
    }

    if (rows >= 2 && cols >= 2) {
        for (r in 0 until rows - 1) {
            for (c in 0 until cols - 1) {

                var a1 = getHorizontalSliceArea(0, r)
                var a2 = getBoundingBoxArea(r + 1, 0, rows - 1, c)
                var a3 = getBoundingBoxArea(r + 1, c + 1, rows - 1, cols - 1)
                minTotalArea = min(minTotalArea, a1 + a2 + a3)

                a1 = getHorizontalSliceArea(r + 1, rows - 1)
                a2 = getBoundingBoxArea(0, 0, r, c)
                a3 = getBoundingBoxArea(0, c + 1, r, cols - 1)
                minTotalArea = min(minTotalArea, a1 + a2 + a3)

                a1 = getVerticalSliceArea(0, c)
                a2 = getBoundingBoxArea(0, c + 1, r, cols - 1)
                a3 = getBoundingBoxArea(r + 1, c + 1, rows - 1, cols - 1)
                minTotalArea = min(minTotalArea, a1 + a2 + a3)

                a1 = getVerticalSliceArea(c + 1, cols - 1)
                a2 = getBoundingBoxArea(0, 0, r, c)
                a3 = getBoundingBoxArea(r + 1, 0, rows - 1, c)
                minTotalArea = min(minTotalArea, a1 + a2 + a3)
            }
        }
    }

    return if (minTotalArea == Int.MAX_VALUE) 0 else minTotalArea
}

fun main() {
    println(
        minimumSum(
            "[[1,0,1],[1,1,1]]".to2DIntArray()
        )
    )
}