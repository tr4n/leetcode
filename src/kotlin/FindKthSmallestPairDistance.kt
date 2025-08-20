package org.example

import kotlin.math.abs

class FindKthSmallestPairDistance(private val data: IntArray) {
    private val n = data.size
    private val tree = Array(4 * n) { emptyList<Int>() }

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = listOf(data[l])
        } else {
            val mid = (l + r) / 2
            build(node * 2, l, mid)
            build(node * 2 + 1, mid + 1, r)
            val left = tree[node * 2]
            val right = tree[node * 2 + 1]
            tree[node] = merge(left, right)
        }
    }

    private fun merge(left: List<Int>, right: List<Int>): List<Int> {
        val m = left.size
        val n = right.size
        var i = 0
        var j = 0
        return buildList {
            while (i < m || j < n) {
                if (i >= m) {
                    add(right[j])
                    j++
                    continue
                }
                if (j >= n) {
                    add(left[i])
                    i++
                    continue
                }
                val a = left[i]
                val b = right[j]
                if (a <= b) {
                    add(a)
                    i++
                } else {
                    add(b)
                    j++
                }
            }
        }

    }

    fun query(left: Int, right: Int, value: Int): Int {
        return query(1, 0, n - 1, value, left, right)
    }


    private fun query(node: Int, l: Int, r: Int, k: Int, i: Int, j: Int): Int {
        if (r < i || l > j) return Int.MAX_VALUE

        if (i <= l && r <= j) {
            val list = tree[node]
            return findSmallestDiff(list, k)
        }

        val mid = (l + r) / 2
        val left = query(node * 2, l, mid, k, i, j)
        val right = query(node * 2 + 1, mid + 1, r, k, i, j)
        return minOf(left, right)
    }

    private fun findSmallestDiff(list: List<Int>, k: Int): Int {
        var l = 0
        var r = list.size - 1
        var result = Int.MAX_VALUE
        while (l <= r) {
            val mid = (l + r) / 2
            if (list[mid] == k) return 0
            result = minOf(result, abs(k - list[mid]))

            if (list[mid] < k) {
                l = mid + 1
            } else {
                r = mid - 1
            }
        }
        return result
    }
}
