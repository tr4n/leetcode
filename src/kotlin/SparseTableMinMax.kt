package org.example

class SparseTableMinMax(private val arr: IntArray) {
    private val n = arr.size
    private val log = IntArray(n + 1)
    private val stMax: Array<IntArray>
    private val stMin: Array<IntArray>

    init {
        for (i in 2..n) log[i] = log[i / 2] + 1

        val k = log[n] + 1
        stMax = Array(n) { IntArray(k) }
        stMin = Array(n) { IntArray(k) }

        for (i in 0 until n) {
            stMax[i][0] = arr[i]
            stMin[i][0] = arr[i]
        }

        for (k in 1 until k) {
            var i = 0
            while (i + (1 shl k) <= n) {
                stMax[i][k] = maxOf(stMax[i][k - 1], stMax[i + (1 shl (k - 1))][k - 1])
                stMin[i][k] = minOf(stMin[i][k - 1], stMin[i + (1 shl (k - 1))][k - 1])
                i++
            }
        }
    }

    fun queryMax(start: Int, end: Int): Int {
        val len = end - start + 1
        val k = log[len]
        return maxOf(stMax[start][k], stMax[end - (1 shl k) + 1][k])
    }

    fun queryMin(start: Int, end: Int): Int {
        val len = end - start + 1
        val k = log[len]
        return minOf(stMin[start][k], stMin[end - (1 shl k) + 1][k])
    }
}
