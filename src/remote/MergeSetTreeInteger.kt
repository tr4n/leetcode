package remote

class MergeSetTree<T : Comparable<T>>(private val arr: List<T>) {
    private val n = arr.size
    private val tree = Array(4 * n) { setOf<T>() }

    init {
        if (arr.isNotEmpty()) build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = setOf(arr[l])
            return
        }
        val mid = (l + r) / 2
        build(node * 2, l, mid)
        build(node * 2 + 1, mid + 1, r)
        tree[node] = tree[node * 2] + tree[node * 2 + 1]
    }

    fun query(ql: Int, qr: Int): Set<T> {
        return query(1, 0, n - 1, ql, qr)
    }

    fun count(ql: Int, qr: Int, onCount: (Set<T>) -> Long): Long {
        return queryCount(1, 0, n - 1, ql, qr, onCount)
    }

    private fun queryCount(node: Int, l: Int, r: Int, ql: Int, qr: Int, onCount: (Set<T>) -> Long): Long {
        if (qr < l || r < ql) return 0
        if (ql <= l && r <= qr) return onCount(tree[node])
        val mid = (l + r) / 2
        val left = queryCount(node * 2, l, mid, ql, qr, onCount)
        val right = queryCount(node * 2 + 1, mid + 1, r, ql, qr, onCount)
        return left + right
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int): Set<T> {
        if (qr < l || r < ql) return emptySet()
        if (ql <= l && r <= qr) return tree[node]
        val mid = (l + r) / 2
        val left = query(node * 2, l, mid, ql, qr)
        val right = query(node * 2 + 1, mid + 1, r, ql, qr)
        return left + right
    }
}

class DiffCountSegmentTree(private val arr: IntArray) {
    private val n = arr.size
    private val tree = IntArray(4 * n)

    init {
        if (arr.isNotEmpty()) build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = 1
            return
        }
        val mid = (l + r) / 2
        build(node * 2, l, mid)
        build(node * 2 + 1, mid + 1, r)
        tree[node] = tree[node * 2] + tree[node * 2 + 1]
    }

    fun countDiff(ql: Int, qr: Int): Int {
        return queryCount(1, 0, n - 1, ql, qr)
    }

    private fun queryCount(node: Int, l: Int, r: Int, ql: Int, qr: Int): Int {
        if (qr < l || r < ql) return 0
        if (ql <= l && r <= qr) return tree[node]
        val mid = (l + r) / 2
        val left = queryCount(node * 2, l, mid, ql, qr)
        val right = queryCount(node * 2 + 1, mid + 1, r, ql, qr)
        return left + right
    }
}

fun subarraysWithKDistinct(nums: IntArray, k: Int): Int {
    val n = nums.size

    fun countDistinct(limit: Int): Int {
        val freq = IntArray(n + 1)
        var l = 0
        var distinctCount = 0
        var cnt = 0
        for (r in 0 until n) {
            val num = nums[r]
            if (freq[num] == 0) distinctCount++
            freq[num]++

            while (distinctCount > limit) {
                val leftMost = nums[l]
                if (freq[leftMost] == 1) distinctCount--
                freq[leftMost]--
                l++
            }
            cnt += (r - l + 1)
        }
        return cnt
    }
    return countDistinct(k) - countDistinct(k - 1)
}

fun countCompleteSubarrays(nums: IntArray): Int {
    val n = nums.max()
    val k = nums.toSet().size

    fun countDistinct(limit: Int): Int {
        val freq = IntArray(n + 1)
        var l = 0
        var distinctCount = 0
        var cnt = 0
        for (r in 0 until nums.size) {
            val num = nums[r]
            if (freq[num] == 0) distinctCount++
            freq[num]++

            while (distinctCount > limit) {
                val leftMost = nums[l]
                if (freq[leftMost] == 1) distinctCount--
                freq[leftMost]--
                l++
            }
            cnt += (r - l + 1)
        }
        return cnt
    }
    return countDistinct(k) - countDistinct(k - 1)
}

fun maxFrequency(nums: IntArray, k: Int, numOperations: Int): Int {
    val n = nums.size
    val maxNum = nums.max()
    val freq = IntArray(maxNum + 1)
    for (num in nums) freq[num]++

    val prefix = IntArray(maxNum + 2)

    for (i in 0..maxNum) {
        prefix[i + 1] = prefix[i] + freq[i]
    }

    fun countBeforeAfter(num: Int): Int {
        val lo = (num - k).coerceAtLeast(0)
        val hi = (num + k).coerceAtMost(maxNum)
        val freqLow = prefix[num] - prefix[lo]
        val freqHigh = (prefix[hi + 1] - prefix[num + 1]).coerceAtLeast(0)
        return freqLow + freqHigh
    }

    var maxFreq = 0

    for (num in 0..maxNum) {
        val beforeAfter = countBeforeAfter(num)
        val f = freq[num] + beforeAfter.coerceAtMost(numOperations)
        maxFreq = maxOf(maxFreq, f)
    }
    return maxFreq
}

fun main() {
    println(
        subarraysWithKDistinct(
            intArrayOf(7, 9, 6, 10, 3, 7, 6, 14, 9, 14, 7, 6, 13, 5),
            4
        )
    )
}