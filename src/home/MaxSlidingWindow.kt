package home

class MaxSlidingWindow(private val data: IntArray) {
    private val n = data.size
    private val tree = IntArray(4 * n)

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = data[l]
        } else {
            val mid = (l + r) / 2
            build(node * 2, l, mid)
            build(node * 2 + 1, mid + 1, r)
            tree[node] = maxOf(tree[node * 2], tree[node * 2 + 1])
        }
    }

    fun getMax(i: Int, j: Int): Int {
        return query(1, 0, n - 1, i, j)
    }

    private fun query(node: Int, l: Int, r: Int, i: Int, j: Int): Int {
        if (r < i || l > j) return Int.MIN_VALUE

        if (i <= l && r <= j) return tree[node]

        val mid = (l + r) / 2
        val left = query(node * 2, l, mid, i, j)
        val right = query(node * 2 + 1, mid + 1, r, i, j)
        return maxOf(left, right)
    }
}

fun maxSlidingWindow(nums: IntArray, k: Int): IntArray {
    val n = nums.size
    val tree = MaxSlidingWindow(nums)
    return IntArray(n - k + 1) {
        tree.getMax(it, it + k - 1)
    }
}