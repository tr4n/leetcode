class PrefixSum1DInt(arr: IntArray) {
    private val prefix = IntArray(arr.size + 1)

    init {
        for (i in arr.indices) prefix[i + 1] = prefix[i] + arr[i]
    }

    fun query(l: Int, r: Int) = prefix[r + 1] - prefix[l]
}

class PrefixSum1DLong(arr: List<Long>) {
    private val prefix = LongArray(arr.size + 1)

    init {
        for (i in arr.indices) prefix[i + 1] = prefix[i] + arr[i]
    }

    fun query(l: Int, r: Int) = prefix[r + 1] - prefix[l]
}

fun rangeSum(nums: IntArray, n: Int, left: Int, right: Int): Int {
    val mod = 1_000_000_007L
    val prefix = LongArray(n + 1)
    for (i in nums.indices) prefix[i + 1] = prefix[i] + nums[i].toLong()
    val sums = mutableListOf<Long>()
    for (i in 0 until n) {
        for (j in i until n) {
            val sum = prefix[j + 1] - prefix[i]
            sums.add(sum)
        }
    }
    sums.sort()
    var result = 0L
    for (i in left - 1 until right) {
        result = (result % mod + sums[i] % mod) % mod
    }
    return (result % mod).toInt()
}


