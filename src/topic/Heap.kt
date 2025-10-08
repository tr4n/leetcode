package topic

import java.util.*

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

fun trapRainWater(heightMap: Array<IntArray>): Int {
    data class Cell(val r: Int, val c: Int, var h: Int = heightMap[r][c])

    val m = heightMap.size
    val n = heightMap[0].size
    val visited = Array(m) { BooleanArray(n) }
    val heap = PriorityQueue<Cell>(compareBy { it.h })

    for (i in 0 until m) {
        if (!visited[i][0]) {
            visited[i][0] = true
            heap.add(Cell(i, 0))
        }
        if (!visited[i][n - 1]) {
            heap.add(Cell(i, n - 1))
            visited[i][n - 1] = true
        }
    }
    for (j in 0 until n) {
        if (!visited[0][j]) {
            visited[0][j] = true
            heap.add(Cell(0, j))
        }
        if (!visited[m - 1][j]) {
            heap.add(Cell(m - 1, j))
            visited[m - 1][j] = true
        }
    }

    val dirX = intArrayOf(0, 0, -1, 1)
    val dirY = intArrayOf(1, -1, 0, 0)

    var total = 0
    while (heap.isNotEmpty()) {
        val (r, c, h) = heap.poll()

        for (i in 0 until 4) {
            val x = r + dirX[i]
            val y = c + dirY[i]
            if (x !in 0 until m || y !in 0 until n) continue
            if (visited[x][y]) continue
            visited[x][y] = true
            val nextHeight = heightMap[x][y]
            if (nextHeight < h) total += h - nextHeight
            heap.add(Cell(x, y, maxOf(h, nextHeight)))
        }
    }
    return total
}