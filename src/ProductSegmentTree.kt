class ProductSegmentTree(private val data: IntArray) {
    private val n = data.size
    private val tree = DoubleArray(4 * n)

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = data[l].toDouble()
        } else {
            val mid = (l + r) / 2
            build(node * 2, l, mid)
            build(node * 2 + 1, mid + 1, r)
            tree[node] = tree[node * 2] * tree[node * 2 + 1]
        }
    }

    fun getProduct(i: Int, j: Int): Double {
        return query(1, 0, n - 1, i, j)
    }

    private fun query(node: Int, l: Int, r: Int, i: Int, j: Int): Double {
        if (r < i || l > j) return 1.0

        if (i <= l && r <= j) return tree[node]

        val mid = (l + r) / 2
        val left = query(node * 2, l, mid, i, j)
        val right = query(node * 2 + 1, mid + 1, r, i, j)
        return left * right
    }
}

fun numSubarrayProductLessThanK(nums: IntArray, k: Int): Int {
    val n = nums.size
    var count = 0
    val tree = ProductSegmentTree(nums)


    // println(product.toList())
    for (i in 0 until n) {
        if (nums[i] >= k) continue
        val start = i
        var end = -1

        var left = i
        var right = n - 1
        while (left <= right) {
            val mid = (left + right) / 2
            if (mid >= n) break
            val prod = tree.getProduct(start, mid)
            if (prod >= k.toDouble()) {
                right = mid - 1
            } else {
                end = mid
                left = mid + 1
            }
        }
        // println("$start $end $target")
        if (end >= start) count += (end - start + 1)
    }

    return count
}

fun main() {
    println(
        numSubarrayProductLessThanK(intArrayOf(10, 9, 10, 4, 3, 8, 3, 3, 6, 2, 10, 10, 9, 3), 19)
    )
}
