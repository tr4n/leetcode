package remote


class UpdateRangeSmallerSegmentTree(arr: IntArray) {
    private val n = arr.size
    private val tree = IntArray(4 * n) { Int.MAX_VALUE }
    private val lazy = IntArray(4 * n) { Int.MAX_VALUE }

    init {
        build(arr, 0, 0, n - 1)
    }

    private fun build(arr: IntArray, node: Int, start: Int, end: Int) {
        if (start == end) {
            tree[node] = arr[start]
            return
        }
        val mid = (start + end) / 2
        build(arr, 2 * node + 1, start, mid)
        build(arr, 2 * node + 2, mid + 1, end)
        tree[node] = minOf(tree[2 * node + 1], tree[2 * node + 2])
    }

    private fun propagate(node: Int, start: Int, end: Int) {
        if (lazy[node] != Int.MAX_VALUE) {
            tree[node] = minOf(tree[node], lazy[node])
            if (start != end) {
                lazy[2 * node + 1] = minOf(lazy[2 * node + 1], lazy[node])
                lazy[2 * node + 2] = minOf(lazy[2 * node + 2], lazy[node])
            }
            lazy[node] = Int.MAX_VALUE
        }
    }

    fun updateRange(l: Int, r: Int, value: Int) {
        updateRange(0, 0, n - 1, l, r, value)
    }

    private fun updateRange(node: Int, start: Int, end: Int, l: Int, r: Int, value: Int) {
        propagate(node, start, end)
        if (start > r || end < l) return
        if (l <= start && end <= r) {
            lazy[node] = minOf(lazy[node], value)
            propagate(node, start, end)
            return
        }
        val mid = (start + end) / 2
        updateRange(2 * node + 1, start, mid, l, r, value)
        updateRange(2 * node + 2, mid + 1, end, l, r, value)
        tree[node] = minOf(tree[2 * node + 1], tree[2 * node + 2])
    }

    fun query(l: Int, r: Int): Int {
        return query(0, 0, n - 1, l, r)
    }

    private fun query(node: Int, start: Int, end: Int, l: Int, r: Int): Int {
        propagate(node, start, end)
        if (start > r || end < l) return Int.MAX_VALUE
        if (l <= start && end <= r) return tree[node]
        val mid = (start + end) / 2
        return minOf(
            query(2 * node + 1, start, mid, l, r),
            query(2 * node + 2, mid + 1, end, l, r)
        )
    }
}