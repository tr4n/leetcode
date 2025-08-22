package remote

class MaxWithIndexSparseTable(private val arr: IntArray) {
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
                st[k][i] = if (arr[left] >= arr[right]) left else right
            }
        }
    }

    fun query(l: Int, r: Int): Pair<Int, Int> {
        val k = log[r - l + 1]
        val left = st[k][l]
        val right = st[k][r - (1 shl k) + 1]
        return if (arr[left] >= arr[right]) arr[left] to left else arr[right] to right
    }
}

class MaxLongWithIndexSparseTable(private val arr: List<Long>) {
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
                st[k][i] = if (arr[left] >= arr[right]) left else right
            }
        }
    }

    fun query(l: Int, r: Int): Pair<Long, Int> {
        val k = log[r - l + 1]
        val left = st[k][l]
        val right = st[k][r - (1 shl k) + 1]
        return if (arr[left] >= arr[right]) arr[left] to left else arr[right] to right
    }
}


fun maxSumOfThreeSubarrays(nums: IntArray, k: Int): IntArray {
    val n = nums.size
    val m = n - k + 1
    val pSums = LongArray(n)
    pSums[0] = nums[0].toLong()
    for (i in 1 until n) pSums[i] = pSums[i - 1] + nums[i].toLong()
    val kSums = List(m) {
        pSums[it + k - 1] - (if (it == 0) 0 else pSums[it - 1])
    }

    val table = MaxLongWithIndexSparseTable(kSums.toList())

    var indexes = intArrayOf(0, k, 2 * k)
    var maxSum = kSums[0] + kSums[k] + kSums[2 * k]
    //  println("n = $n, k = $k, m = $m, m-k=${m - k}")
    for (i in k until m - k) {
        val midSum = kSums[i]
        val (leftSum, leftIndex) = table.query(0, i - k)
        val (rightSum, rightIndex) = table.query(i + k, m - 1)
        val totalSum = leftSum + midSum + rightSum
        ///  println("$leftIndex,$i,$rightIndex: $totalSum")
        if (totalSum > maxSum) {
            maxSum = totalSum
            indexes = intArrayOf(leftIndex, i, rightIndex)
        }
    }
    return indexes
}

fun main() {
    println(
        maxSumOfThreeSubarrays(
            intArrayOf(18, 11, 14, 7, 16, 3, 18, 11, 3, 8), 3
        ).toList()
    )
}

