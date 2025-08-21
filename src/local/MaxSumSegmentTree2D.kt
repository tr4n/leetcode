package local

import java.util.*
import kotlin.collections.iterator

class MaxLongSparseTable(private val arr: List<Long>) {
    private val n = arr.size
    private val log = IntArray(n + 1)
    private val stMax: Array<LongArray>

    init {
        for (i in 2..n) log[i] = log[i / 2] + 1

        val k = log[n] + 1
        stMax = Array(n) { LongArray(k) }

        for (i in 0 until n) {
            stMax[i][0] = arr[i]
        }

        for (k in 1 until k) {
            var i = 0
            while (i + (1 shl k) <= n) {
                stMax[i][k] = minOf(stMax[i][k - 1], stMax[i + (1 shl (k - 1))][k - 1])
                i++
            }
        }
    }

    fun queryMax(start: Int, end: Int): Long {
        val len = end - start + 1
        val k = log[len]
        return maxOf(stMax[start][k], stMax[end - (1 shl k) + 1][k])
    }
}

class MonotoneDescendingMap {
    private val map = TreeMap<Long, Long>()

    fun insert(y: Long, value: Long) {
        val existing = map[y]
        if (existing != null && existing >= value) {
            return
        }
        map[y] = value

        val toRemove = mutableListOf<Long>()
        var lower = map.lowerEntry(y)
        while (lower != null && lower.value <= value) {
            toRemove.add(lower.key)
            lower = map.lowerEntry(lower.key)
        }
        for (k in toRemove) map.remove(k)
    }

    fun getMap(): TreeMap<Long, Long> = map
}


class SumMax2DSegmentTree(val pairs: List<Pair<Long, Long>>) {
    data class Node(
        val pairs: List<Pair<Long, Long>> = emptyList(),
        val map: MonotoneDescendingMap = MonotoneDescendingMap()
    )

    private val n = pairs.size
    private val tree = Array(4 * n) { Node() }

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            val (x, y) = pairs[l]
            val pairs = listOf(x to y)
            val map = MonotoneDescendingMap()
            map.insert(y, x + y)
            tree[node] = Node(pairs, map)
            return
        }
        val mid = (l + r) / 2
        build(2 * node, l, mid)
        build(2 * node + 1, mid + 1, r)
        val left = tree[2 * node]
        val right = tree[2 * node + 1]
        tree[node] = merge(left, right)
    }

    private fun merge(left: Node, right: Node): Node {
        val pairs = mutableListOf<Pair<Long, Long>>()
        val a = left.pairs
        val b = right.pairs
        var i = 0
        var j = 0
        while (i < a.size && j < b.size) {
            if (a[i].first <= b[j].first) {
                pairs.add(a[i++])
            } else {
                pairs.add(b[j++])
            }
        }
        while (i < a.size) pairs.add(a[i++])
        while (j < b.size) pairs.add(b[j++])

        val map = MonotoneDescendingMap()
        for ((x, y) in pairs) {
            map.insert(y, x + y)
        }
        return Node(pairs, map)
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int, x: Int, y: Int): Long {
        if (l > qr || r < ql) return -1
        if (ql <= l && r <= qr) {
            return calculateMax(tree[node], x, y)
        }
        val mid = (l + r) / 2
        val left = query(2 * node, l, mid, ql, qr, x, y)
        val right = query(2 * node + 1, mid + 1, r, ql, qr, x, y)
        return maxOf(left, right)
    }

    private fun calculateMax(
        node: Node,
        minX: Int,
        minY: Int
    ): Long {
        val (pairs, map) = node
        //   println(pairs)
        val fromIndex = findFirstGreaterIndex(pairs, 0, minX.toLong()) { it.first }
        if (fromIndex == -1) return -1
        var ansMax = -1L


        val sub = map.getMap().tailMap(minY.toLong(), true)
        for ((y, sum) in sub) {
            if (sum - y >= minX) {
                return maxOf(ansMax, sum)
            }
        }

//        for (i in fromIndex until n) {
//            val (x, y) = pairs[i]
//            if (y >= minY) {
//                 ansMax = maxOf(ansMax, x + y)
//            }
//        }
        return ansMax
    }

    fun getMaxSum(start: Int, end: Int, x: Int, y: Int): Long {
        return query(1, 0, n - 1, start, end, x, y)
    }

    private fun <T> findFirstGreaterIndex(list: List<T>, fromIndex: Int, k: Long, value: (T) -> Long): Int {
        if (list.isEmpty()) return -1
        var l = fromIndex
        var r = list.size - 1
        var result = -1
        while (l <= r) {
            val mid = (l + r) / 2
            if (value(list[mid]) >= k) {
                result = mid
                r = mid - 1
            } else l = mid + 1
        }
        return result
    }

    private fun <T> findFirstSmallerIndex(list: List<T>, fromIndex: Int, k: Int, value: (T) -> Int): Int {
        var l = fromIndex
        var r = list.size - 1
        var result = -1
        while (l <= r) {
            val mid = (l + r) / 2
            if (value(list[mid]) < k) {
                result = mid
                l = mid + 1
            } else r = mid - 1
        }
        return result
    }
}

fun maximumSumQueries(nums1: IntArray, nums2: IntArray, queries: Array<IntArray>): IntArray {

    val minX = queries.minOf { it[0] }
    val minY = queries.minOf { it[1] }
    val pairs = (0 until nums1.size).filter {
        nums1[it] >= minX && nums2[it] >= minY
    }.map { nums1[it].toLong() to nums2[it].toLong() }
    if (pairs.isEmpty()) {
        return IntArray(queries.size) { -1 }
    }
    val tree = SumMax2DSegmentTree(pairs)

    return IntArray(queries.size) {
        val (x, y) = queries[it]
        tree.getMaxSum(0, pairs.size - 1, x, y).toInt()
    }
}

fun main() {
    println(
        maximumSumQueries(
            intArrayOf(4, 3, 1, 2),
            intArrayOf(2, 4, 9, 5),
            arrayOf(intArrayOf(4, 1), intArrayOf(1, 3), intArrayOf(2, 5))
        ).toList()
    )
}