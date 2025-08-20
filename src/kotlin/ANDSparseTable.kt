package org.example

import kotlin.concurrent.atomics.AtomicArray
import kotlin.math.abs

class ANDSparseTable(private val arr: IntArray) {
    private val n = arr.size
    private val log = IntArray(n + 1)
    private val stMax: Array<IntArray>

    init {
        for (i in 2..n) log[i] = log[i / 2] + 1

        val k = log[n] + 1
        stMax = Array(n) { IntArray(k) }

        for (i in 0 until n) {
            stMax[i][0] = arr[i]
        }

        for (k in 1 until k) {
            var i = 0
            while (i + (1 shl k) <= n) {
                stMax[i][k] = (stMax[i][k - 1] and stMax[i + (1 shl (k - 1))][k - 1])
                i++
            }
        }
    }

    fun queryAND(start: Int, end: Int): Int {
        val len = end - start + 1
        val k = log[len]
        return (stMax[start][k] and stMax[end - (1 shl k) + 1][k])
    }
}

fun closestToTarget(arr: IntArray, target: Int): Int {
    val n = arr.size
    val table = ANDSparseTable(arr)
    var result = abs(table.queryAND(0, n - 1) - target)

    for (i in 0 until n) {
        var l = i
        var r = n - 1

        while (l <= r) {
            val mid = l + (r - l) / 2
            val value = table.queryAND(i, mid)
            //   println("$i-$mid : $value")
            if (value == target) return 0
            result = minOf(result, abs(value - target))

            if (value > target) {
                l = mid + 1
            } else {
                r = mid - 1
            }
        }

        for (j in listOf(l, l - 1, l + 1)) {
            if (j in i until n) {
                val value = table.queryAND(i, j)
                result = minOf(result, abs(value - target))
            }
        }
    }
    return result
}

fun countSubarrays(nums: IntArray, k: Int): Long {
    val n = nums.size
    val table = ANDSparseTable(nums)

    var cnt = 0L
    for (i in 0 until n) {
        var l = i
        var r = n - 1
        var endIndex = -1
        while (l <= r) {
            val mid = (l + r) / 2
            val value = table.queryAND(i, mid)
            if (value >= k) {
                if (value == k) endIndex = mid
                l = mid + 1
            } else {
                r = mid - 1
            }
        }

        l = i
        r = n - 1
        var startIndex = -1

        while (l <= r) {
            val mid = (l + r) / 2
            val value = table.queryAND(i, mid)
            if (value <= k) {
                if (value == k) startIndex = mid
                r = mid - 1
            } else {
                l = mid + 1
            }
        }
        // println("$i $endIndex $startIndex")
        if (startIndex >= i && endIndex >= i && endIndex >= startIndex) {
            cnt += (endIndex - startIndex + 1).toLong()
        }
    }
    return cnt
}

fun splitArray(nums: IntArray, k: Int): Int {
    val n = nums.size
    val sums = IntArray(n)
    sums[0] = nums[0]
    for (i in 1 until n) {
        sums[i] = sums[i - 1] + nums[i]
    }

    val dp = Array(k) { IntArray(n) }
    for (j in 0 until n) {
        dp[0][j] = sums[j]
    }

    for (p in 1 until k) {

        for (i in 1 until n) {
            var d = Int.MAX_VALUE
            for (j in 0 until i) {
                d = maxOf(sums[i] - sums[j], dp[p-1][j]).coerceAtMost(d)
            }
            dp[p][i] = d
        }
    }
    println(dp.joinToString("\n") { it.toList().toString() })
    return dp[k-1][n-1]
}

fun main() {
    println(
        splitArray(intArrayOf(1,2,3,4,5), 2)
    )
}

