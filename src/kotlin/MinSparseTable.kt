package org.example

import kotlin.math.min

class MinSparseTable(private val arr: IntArray) {
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
                stMax[i][k] = minOf(stMax[i][k - 1], stMax[i + (1 shl (k - 1))][k - 1])
                i++
            }
        }
    }

    fun queryMin(start: Int, end: Int): Int {
        val len = end - start + 1
        val k = log[len]
        return min(stMax[start][k], stMax[end - (1 shl k) + 1][k])
    }
}

