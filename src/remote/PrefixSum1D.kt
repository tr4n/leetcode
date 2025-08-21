
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
