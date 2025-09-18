package topic

import java.util.PriorityQueue

fun maxSum(grid: Array<IntArray>, limits: IntArray, k: Int): Long {
    val m = grid.size
    val n = grid[0].size
    grid.forEach { it.sortDescending() }

    val heap = PriorityQueue<Int>()
    for (i in 0 until m) {
        for (j in 0 until minOf(n, limits[i])) {
            heap.add(grid[i][j])
            if (heap.size > k) heap.poll()
        }
    }
    return heap.sumOf { it.toLong() }
}