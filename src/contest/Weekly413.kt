package contest

import local.to2DIntArray
import java.util.*
import kotlin.math.abs

class Weekly413 {
    fun resultsArray(queries: Array<IntArray>, k: Int): IntArray {
        val n = queries.size
        val pq = PriorityQueue<IntArray>(compareByDescending { abs(it[0]) + abs(it[1]) })
        val result = IntArray(n) { -1 }
        for (i in 0 until n) {
            val query = queries[i]
            pq.add(query)
            if (pq.size < k) continue
            if (pq.size > k) pq.poll()
            val peek = pq.peek()
            result[i] = abs(peek[0]) + abs(peek[1])
        }
        return result
    }

    fun maxScore2(grid: List<List<Int>>): Int {
        val m = grid.size
        val n = grid[0].size
        val size = m * n
        val list = grid.flatMapIndexed { index, list ->
            list.map { it to index }
        }.sortedBy { it.first }

        val maxBit = 1 shl m
        val dp = Array(size + 1) { IntArray(maxBit) }

        var maxSum = 0
        for (i in 1..size) {
            val (num, row) = list[i - 1]
            val rowBit = 1 shl row
            for (j in (i - 1) downTo 0) {
                for (bit in 0 until maxBit) {
                    if (bit and rowBit != 0) continue
                    val newBit = bit or rowBit
                    dp[i][newBit] = maxOf(dp[i][newBit], dp[j][bit] + num)
                    maxSum = maxOf(maxSum, dp[i][newBit])
                }
            }
        }
        println(grid.joinToString("\n"))
        return maxSum
    }

    fun maxScore(grid: List<List<Int>>): Int {
        val m = grid.size
        val n = grid[0].size

        val matrix = grid.map { it.sortedDescending() }
        val used = BooleanArray(101)

        val remainingMax = IntArray(m + 1)
        for (i in (m - 1) downTo 0) {
            remainingMax[i] = remainingMax[i + 1] + matrix[i].first()
        }

        var maxSum = 0
        fun dfs(row: Int, sum: Int) {
            if (row == m) {
                maxSum = maxOf(maxSum, sum)
                return
            }
            if (sum + remainingMax[row] <= maxSum) return

            for (num in matrix[row]) {
                if (used[num]) continue
                used[num] = true
                dfs(row + 1, sum + num)
                used[num] = false
            }
            dfs(row + 1, sum)
        }
        dfs(0, 0)
        return maxSum
    }
}

fun main() {
    val week = Weekly413()
    println(
        week.maxScore(
            "[[92,11,45,88,38,13,65,85],[52,83,3,14,82,51,27,59],[65,69,99,27,7,70,39,43],[43,46,22,19,75,70,57,50],[54,36,91,80,74,43,62,61],[35,45,19,32,92,50,93,88],[60,15,93,10,89,32,51,11],[82,66,42,61,78,94,66,7],[75,56,49,78,81,61,79,50]]".to2DIntArray()
                .map { it.toList() })
    )
}