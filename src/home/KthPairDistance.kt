package home

import kotlin.math.abs

class KthPairDistance(private val data: IntArray) {
    private val n = data.size
    private val tree = Array(4 * n) { emptyList<Int>() }
    var minDistance = Int.MAX_VALUE
        private set
    var maxDistance = Int.MIN_VALUE
        private set

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
                    add(right[j++])
                    continue
                }
                if (j >= n) {
                    add(left[i++])
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
                val distance = abs(a - b)
                minDistance = minOf(minDistance, distance)
                maxDistance = maxOf(maxDistance, distance)
            }
        }

    }

    fun countInRange(i: Int, j: Int, low: Int, high: Int): Int {
        return query(1, 0, n - 1, i, j, low, high)
    }

    private fun query(node: Int, l: Int, r: Int, i: Int, j: Int, low: Int, high: Int): Int {
        if (r < i || l > j) return 0

        if (i <= l && r <= j) {
            val less = countLessThan(tree[node], low)
            val greater = countGreaterThan(tree[node], high)
            val total = tree[node].size
            return total - less - greater
        }

        val mid = (l + r) / 2
        val left = query(node * 2, l, mid, i, j, low, high)
        val right = query(node * 2 + 1, mid + 1, r, i, j, low, high)
        return left + right
    }

    private fun countLessThan(list: List<Int>, x: Int): Int {
        var l = 0
        var r = list.size
        while (l < r) {
            val mid = (l + r) / 2
            if (list[mid] < x) l = mid + 1 else r = mid
        }
        return l
    }

    private fun countGreaterThan(list: List<Int>, x: Int): Int {
        var l = 0
        var r = list.size
        while (l < r) {
            val mid = (l + r) / 2
            if (list[mid] <= x) l = mid + 1 else r = mid
        }
        return list.size - l
    }
}

fun smallestDistancePair(nums: IntArray, k: Int): Int {
    val n = nums.size
    val tree = KthPairDistance(nums)

    fun countLessThan(x: Int): Int {
        var cnt = 0
        for (i in 0 until n) {
            val num = nums[i]
            cnt += tree.countInRange(i + 1, n - 1, num - x, num + x)
        }
        return cnt
    }

    var low = tree.minDistance
    var high = tree.maxDistance
    println("low: $low, $high: $high")
    while (low < high) {
        val mid = low + (high - low) / 2
        val count = countLessThan(mid)
        //   println("$mid $count")
        if (count < k) {
            low = mid + 1
        } else {
            high = mid
        }
    }
    return low
}

fun main(){
    println(
        smallestDistancePair(
            intArrayOf(1,6,1),
            3
        )
    )
}