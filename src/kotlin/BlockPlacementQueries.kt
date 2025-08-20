package org.example

import java.util.*
import kotlin.math.*

class BlockPlacementQueries(private val n: Int) {
    private val tree = IntArray(4 * n)

    fun getMax(start: Int, end: Int): Int {
        return query(1, 0, n - 1, start, end)
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int): Int {
        if (l > qr || r < ql) return 0

        if (ql <= l && r <= qr) return tree[node]

        val mid = (l + r) / 2
        val left = query(2 * node, l, mid, ql, qr)
        val right = query(2 * node + 1, mid + 1, r, ql, qr)

        return max(left, right)
    }

    fun update(x: Int, value: Int) {
        update(1, 0, n - 1, x, x, value)
    }

    private fun update(node: Int, l: Int, r: Int, ul: Int, ur: Int, value: Int) {
        if (l > ur || r < ul) return

        if (l == r) {
            //   println("node $node ${tree[node]} -> $value")
            tree[node] = value
            return
        }

        val mid = (l + r) / 2
        update(2 * node, l, mid, ul, ur, value)
        update(2 * node + 1, mid + 1, r, ul, ur, value)
        tree[node] = max(tree[2 * node], tree[2 * node + 1])
    }
}

fun getResults(queries: Array<IntArray>): List<Boolean> {
    val queryCount = queries.size
    val n = min(50_000, 3 * queryCount)
    val segmentTree = BlockPlacementQueries(n)
    val treeMap = TreeMap<Int, Int>()
    segmentTree.update(0, n)
    treeMap[0] = n

    val answers = mutableListOf<Boolean>()
    for (query in queries) {
        when (query[0]) {
            1 -> {
                val obstacle = query[1]
                val nextObstacle = treeMap.ceilingKey(obstacle) ?: n
                val previousObstacle = treeMap.floorKey(obstacle) ?: 0

                //  println("$obstacle ($previousObstacle, $nextObstacle)")
                segmentTree.update(previousObstacle, obstacle - previousObstacle)
                segmentTree.update(obstacle, nextObstacle - obstacle)

                treeMap[previousObstacle] = obstacle - previousObstacle
                treeMap[obstacle] = nextObstacle - obstacle
                //   println(treeMap.toList())
            }

            2 -> {
                val end = query[1]
                val size = query[2]
                val obstacle = treeMap.floorKey(end) ?: 0
                val d = treeMap[obstacle] ?: n

                segmentTree.update(obstacle, end - obstacle)
                val isAbleToPlace = segmentTree.getMax(0, obstacle) >= size
                segmentTree.update(obstacle, d)
                answers.add(isAbleToPlace)
            }
        }
    }
    return answers
}

fun estimateLevel(n: Int): Int {
    val a = 1.0
    val b = 3.0
    val c = 2.0
    val d = -6.0 * n

    // Using Cardano's formula to solve cubic: x^3 + 3x^2 + 2x - 6n = 0
    val f = 1.0 * ((3.0 * c / a) - (b * b) / (a * a)) / 3.0
    val g = 1.0 * ((2.0 * b * b * b) / (a * a * a) - (9.0 * b * c) / (a * a) + (27.0 * d) / a) / 27
    val h = 1.0 * g * g / 4.0 + f * f * f / 27.0

    val x: Double = if (h <= 0.0) {
        // Three real roots
        val i = sqrt(g * g / 4.0 - h)
        val j = cbrt(i)
        val k = acos(-g / (2.0 * i))
        val m = cos(k / 3.0)
        val root = 2.0 * j * m - b / (3.0 * a)
        root
    } else {
        // One real root
        val r = -g / 2 + sqrt(h)
        val s = cbrt(r)
        val t = -g / 2 - sqrt(h)
        val u = cbrt(t)
        s + u - b / (3 * a)
    }

    val level = floor(x).toInt()
    return level
}

fun calculateLevel(n: Int): Int {
    if (n <= 3) return 1
    val target = 6L * n
    var left = 1
    var right = 2000

    var level = 0
    while (left <= right) {
        val x = (left + right) / 2
        val totalBoxes = x.toLong() * (x + 1L) * (x + 2L)

        if (totalBoxes <= target) {
            level = x
            left = x + 1
        } else {
            right = x - 1
        }
    }
    return level
}

fun minimumBoxes(n: Int): Int {
    if (n <= 3) return n

    val level = calculateLevel(n)
    val usedBoxes = (level.toLong() * (level + 1L) * (level + 2L)) / 6L
    val floorBoxes = (level.toLong() * (level + 1L)) / 2

    val remaining = n - usedBoxes
    val extra = ceil(sqrt(2.0 * remaining + 0.25) - 0.5).toInt()


    return (floorBoxes + extra).toInt()
}

fun maxDistance(colors: IntArray): Int {
    val n = colors.size
    if (n < 2) return 0
    if (n == 2) return if (colors[0] == colors[1]) 0 else 1

    var maxDistance = 0
    for (i in 0 until n - 1) {
        for (j in (n - 1) downTo (i + 1)) {
            if (colors[j] != colors[i] && j - i  > maxDistance) {
                maxDistance = j - i
            }
        }
    }
    return maxDistance
}


fun main() {
    println(
        minimumBoxes(694251486)
    )
}
