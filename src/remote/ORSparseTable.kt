package remote

import kotlin.math.abs

class ORSparseTable(private val arr: IntArray) {
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
                stMax[i][k] = (stMax[i][k - 1] or stMax[i + (1 shl (k - 1))][k - 1])
                i++
            }
        }
    }

    fun queryOR(start: Int, end: Int): Int {
        val len = end - start + 1
        val k = log[len]
        return (stMax[start][k] or stMax[end - (1 shl k) + 1][k])
    }
}

fun minimumDifference(nums: IntArray, k: Int): Int {
    val n = nums.size
    val table = ORSparseTable(nums)

    var minDist = Int.MAX_VALUE
    for (i in 0 until n) {
        var l = i
        var r = n - 1

        while (l <= r) {
            val mid = (l + r) / 2
            val value = table.queryOR(i, mid)
            if (value == k) return 0
            if (value < k) {
                l = mid + 1
            } else {
                r = mid - 1
            }
        }

        for (pos in listOf(l - 1, l)) {
           // println("$i $pos")
            if (pos !in 0 until n) continue
            val value = table.queryOR(i, pos)
            minDist = minOf(minDist, abs(value - k))
        }
    }
    return minDist
}

fun main() {
    println(
        minimumDifference(intArrayOf(1), 10)
    )
}

