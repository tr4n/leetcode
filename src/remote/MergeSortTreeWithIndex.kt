package remote

class MergeSortTreeWithIndex(private val arr: IntArray) {
    private val n = arr.size
    private val tree = Array(4 * n) { mutableListOf<Pair<Int, Int>>() }

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = mutableListOf(arr[l] to l) // (giá trị, index gốc)
            return
        }
        val mid = (l + r) / 2
        build(node * 2, l, mid)
        build(node * 2 + 1, mid + 1, r)
        tree[node] = merge(tree[node * 2], tree[node * 2 + 1])
    }

    private fun merge(a: List<Pair<Int, Int>>, b: List<Pair<Int, Int>>): MutableList<Pair<Int, Int>> {
        val result = mutableListOf<Pair<Int, Int>>()
        var i = 0
        var j = 0
        while (i < a.size && j < b.size) {
            if (a[i].first <= b[j].first) {
                result.add(a[i++])
            } else {
                result.add(b[j++])
            }
        }
        while (i < a.size) result.add(a[i++])
        while (j < b.size) result.add(b[j++])
        return result
    }

    fun query(ql: Int, qr: Int): List<Pair<Int, Int>> {
        return query(1, 0, n - 1, ql, qr)
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int): List<Pair<Int, Int>> {
        if (qr < l || r < ql) return emptyList()
        if (ql <= l && r <= qr) return tree[node]
        val mid = (l + r) / 2
        val left = query(node * 2, l, mid, ql, qr)
        val right = query(node * 2 + 1, mid + 1, r, ql, qr)
        return merge(left, right)
    }
}
