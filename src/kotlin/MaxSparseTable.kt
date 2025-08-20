package org.example

class MaxSparseTable(private val arr: List<Int>) {
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
                stMax[i][k] = maxOf(stMax[i][k - 1], stMax[i + (1 shl (k - 1))][k - 1])
                i++
            }
        }
    }

    fun queryMax(start: Int, end: Int): Int {
        val len = end - start + 1
        val k = log[len]
        return maxOf(stMax[start][k], stMax[end - (1 shl k) + 1][k])
    }
}

fun findPrefixScore(nums: IntArray): LongArray {
    val n = nums.size
    var preScore = 0L
    var preMax = -1L
    return LongArray(n) {
        val num = nums[it].toLong()
        val newMax = maxOf(preMax, num)
        val conversation = num + newMax
        val score = preScore + conversation
        preMax = newMax
        preScore = score
        score
    }
}

fun maximumBeauty(items: Array<IntArray>, queries: IntArray): IntArray {
    items.sortBy { it[0] }
    val n = items.size
    val table = MaxSparseTable(items.map { it[1] })
    val answers = IntArray(queries.size)
    for (i in queries.indices) {
        val target = queries[i]
        var l = 0
        var r = n - 1
        var endIndex = -1
        while (l <= r) {
            val mid = (l + r) / 2
            val value = items[mid].first()
            if (value <= target) {
                endIndex = mid
                l = mid + 1
            } else {
                r = mid - 1
            }
        }
        if (endIndex < 0) continue
        answers[i] = table.queryMax(0, endIndex)
    }
    return answers
}

