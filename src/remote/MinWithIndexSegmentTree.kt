package remote

class MinWithIndexSegmentTree(private val arr: IntArray) {
    private val n = arr.size
    private val tree = Array(4 * n) { Pair(Int.MAX_VALUE, -1) }

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = arr[l] to l
            return
        }
        val mid = (l + r) / 2
        build(node * 2, l, mid)
        build(node * 2 + 1, mid + 1, r)
        tree[node] = minOf(tree[node * 2], tree[node * 2 + 1])
    }

    private fun minOf(a: Pair<Int, Int>, b: Pair<Int, Int>): Pair<Int, Int> {
        return if (a.first <= b.first) a else b
    }

    fun query(ql: Int, qr: Int): Pair<Int, Int> {
        return query(1, 0, n - 1, ql, qr)
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int): Pair<Int, Int> {
        if (qr < l || r < ql) return Int.MAX_VALUE to -1
        if (ql <= l && r <= qr) return tree[node]
        val mid = (l + r) / 2
        val left = query(node * 2, l, mid, ql, qr)
        val right = query(node * 2 + 1, mid + 1, r, ql, qr)
        return minOf(left, right)
    }

    fun update(idx: Int, value: Int) {
        update(1, 0, n - 1, idx, value)
    }

    private fun update(node: Int, l: Int, r: Int, idx: Int, value: Int) {
        if (l == r) {
            tree[node] = value to idx
            return
        }
        val mid = (l + r) / 2
        if (idx <= mid) update(node * 2, l, mid, idx, value)
        else update(node * 2 + 1, mid + 1, r, idx, value)
        tree[node] = minOf(tree[node * 2], tree[node * 2 + 1])
    }
}
