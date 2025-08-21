package local

class NumSubarrayBoundedMax(private val arr: IntArray) {
    private val n = arr.size
    private val log = IntArray(n + 1)
    private val st: Array<IntArray>

    init {
        // log2
        for (i in 2..n) {
            log[i] = log[i / 2] + 1
        }

        val K = log[n] + 1
        st = Array(n) { IntArray(K) }

        // base
        for (i in 0 until n) st[i][0] = arr[i]

        // build
        for (j in 1 until K) {
            var i = 0
            while (i + (1 shl j) <= n) {
                st[i][j] = maxOf(st[i][j - 1], st[i + (1 shl (j - 1))][j - 1])
                i++
            }
        }
    }

    // Query OR [l,r] in O(1)
    fun query(l: Int, r: Int): Int {
        val j = log[r - l + 1]
        return maxOf(st[l][j], st[r - (1 shl j) + 1][j])
    }
}


fun numberOfSubarrays(nums: IntArray): Long {
    val n = nums.size
    val table = NumSubarrayBoundedMax(nums)
    val numToIndexes = mutableMapOf<Int, MutableList<Int>>()

    var cnt = 0L
    for (i in 0 until n) {
        val num = nums[i]
        val indexes = numToIndexes[num]
        if (indexes.isNullOrEmpty()) {
            numToIndexes[num] = mutableListOf(i)
            continue
        }
        val lastIndex = indexes.last()
        val max = table.query(lastIndex, i)
        if (max == num) {
            numToIndexes[num]?.add(i)
            continue
        }
        val size = indexes.size.toLong()
        cnt += ((size * (size + 1L)) / 2)
        numToIndexes[num] = mutableListOf(i)
    }

    for (indexes in numToIndexes.values) {
        val size = indexes.size.toLong()
        cnt += ((size * (size + 1L)) / 2)
    }
    return cnt
}


fun numSubarrayBoundedMax(nums: IntArray, left: Int, right: Int): Int {
    val n = nums.size
    val table = NumSubarrayBoundedMax(nums)

    var cnt = 0
    for (i in 0 until n) {
        var l = i
        var r = n - 1
        var startIndex = -1
        while (l <= r) {
            val mid = (l + r) / 2
            val max = table.query(i, mid)
            if (max >= left) {
                startIndex = mid
                r = mid - 1
            } else {
                l = mid + 1
            }
        }
        l = i
        r = n - 1
        var endIndex = -1
        while (l <= r) {
            val mid = (l + r) / 2
            val max = table.query(i, mid)
            if (max <= right) {
                endIndex = mid
                l = mid + 1
            } else {
                r = mid - 1
            }
        }
        if (startIndex >= 0 && endIndex >= 0) {
            cnt += (endIndex - startIndex + 1)
        }
    }
    return cnt
}