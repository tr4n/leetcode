package org.example

class CountGreaterFenwick(private val values: List<Int>) {
    private val sorted = values.distinct().sorted()
    private val index = sorted.withIndex().associate { it.value to it.index + 1 }
    private val tree = IntArray(sorted.size + 1)

    private fun lsb(x: Int) = x and -x

    fun update(i: Int, delta: Int) {
        var idx = i
        while (idx < tree.size) {
            tree[idx] += delta
            idx += lsb(idx)
        }
    }

    fun prefixSum(i: Int): Int {
        var idx = i
        var sum = 0
        while (idx > 0) {
            sum += tree[idx]
            idx -= lsb(idx)
        }
        return sum
    }

    fun countGE(x: Int): Int {
        val pos = sorted.binarySearch(x).let { if (it < 0) -it - 1 else it }
        if (pos == sorted.size) return 0
        val idx = pos + 1
        val total = prefixSum(sorted.size)
        val less = prefixSum(idx - 1)
        return total - less
    }
}
