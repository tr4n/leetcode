package local

class SparseTableOR(private val arr: IntArray) {
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
                st[i][j] = st[i][j - 1] or st[i + (1 shl (j - 1))][j - 1]
                i++
            }
        }
    }

    // Query OR [l,r] in O(1)
    fun query(l: Int, r: Int): Int {
        val j = log[r - l + 1]
        return st[l][j] or st[r - (1 shl j) + 1][j]
    }
}

fun minimumSubarrayLength(nums: IntArray, k: Int): Int {
    if (nums.any { it >= k }) return 1

    val n = nums.size
    val sparseTable = SparseTableOR(nums)

    var minLength = n + 1
    for (i in 0 until n - 1) {
        var l = i + 1
        var r = n - 1
        var ans = -1
        while (l <= r) {
            val mid = (l + r) / 2
            val orValue = sparseTable.query(i, mid)
            if (orValue >= k) {
                ans = mid
                r = mid - 1
            } else {
                l = mid + 1
            }
        }
        if (ans >= 0) {
            minLength = minOf(minLength, ans - i + 1)
        }
    }
    return if (minLength > n) -1 else minLength
}