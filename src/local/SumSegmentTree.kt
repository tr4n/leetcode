package local

class SumSegmentTree(nums: IntArray) {
    private val n = nums.size
    private val data = nums
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
            tree[node] = tree[node * 2] + tree[node * 2 + 1]
        }
    }


    private fun query(node: Int, l: Int, r: Int, i: Int, j: Int): Int {
        if (r < i || l > j) return 0

        if (i <= l && r <= j) return tree[node]

        val mid = (l + r) / 2
        val left = query(node * 2, l, mid, i, j)
        val right = query(node * 2 + 1, mid + 1, r, i, j)
        return left + right
    }

    private fun update(node: Int, l: Int, r: Int, idx: Int, value: Int) {
        if (l == r) {
            tree[node] = value
        } else {
            val mid = (l + r) / 2
            if (idx <= mid) {
                update(node * 2, l, mid, idx, value)
            } else {
                update(node * 2 + 1, mid + 1, r, idx, value)
            }
            tree[node] = tree[node * 2] + tree[node * 2 + 1]
        }
    }

    fun update(index: Int, `val`: Int) {
        update(1, 0, n - 1, index, `val`)
    }

    fun sumRange(left: Int, right: Int): Int {
        return query(1, 0, n - 1, left, right)
    }

}


class SumLongSegmentTree(nums: List<Long>) {
    private val n = nums.size
    private val data = nums
    private val tree = LongArray(4 * n)

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
            tree[node] = tree[node * 2] + tree[node * 2 + 1]
        }
    }


    private fun query(node: Int, l: Int, r: Int, i: Int, j: Int): Long {
        if (r < i || l > j) return 0

        if (i <= l && r <= j) return tree[node]

        val mid = (l + r) / 2
        val left = query(node * 2, l, mid, i, j)
        val right = query(node * 2 + 1, mid + 1, r, i, j)
        return left + right
    }

    private fun update(node: Int, l: Int, r: Int, idx: Int, value: Long) {
        if (l == r) {
            tree[node] = value
        } else {
            val mid = (l + r) / 2
            if (idx <= mid) {
                update(node * 2, l, mid, idx, value)
            } else {
                update(node * 2 + 1, mid + 1, r, idx, value)
            }
            tree[node] = tree[node * 2] + tree[node * 2 + 1]
        }
    }

    fun update(index: Int, value: Long) {
        update(1, 0, n - 1, index, value)
    }

    fun sumRange(left: Int, right: Int): Long {
        return query(1, 0, n - 1, left, right)
    }

}