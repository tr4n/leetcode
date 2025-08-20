package org.example

class MinWithIndexSparseTable(private val arr: IntArray) {
    private val n = arr.size
    private val log = IntArray(n + 1)
    private val st: Array<IntArray>

    init {
        for (i in 2..n) log[i] = log[i / 2] + 1

        val K = log[n] + 1
        st = Array(K) { IntArray(n) }

        for (i in 0 until n) st[0][i] = i

        for (k in 1 until K) {
            val len = 1 shl k
            val half = len shr 1
            for (i in 0..n - len) {
                val left = st[k - 1][i]
                val right = st[k - 1][i + half]
                st[k][i] = if (arr[left] <= arr[right]) left else right
            }
        }
    }

    fun query(l: Int, r: Int): Pair<Int, Int> {
        val k = log[r - l + 1]
        val left = st[k][l]
        val right = st[k][r - (1 shl k) + 1]
        return if (arr[left] <= arr[right]) arr[left] to left else arr[right] to right
    }
}
