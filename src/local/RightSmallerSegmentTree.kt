package local

class RightSmallerSegmentTree(private val data: IntArray) {
    private val n = data.size
    private val tree = Array(4 * n) { emptyList<Int>() }

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = listOf(data[l])
        } else {
            val mid = (l + r) / 2
            build(node * 2, l, mid)
            build(node * 2 + 1, mid + 1, r)
            val left = tree[node * 2]
            val right = tree[node * 2 + 1]
            tree[node] = merge(left, right)
        }
    }

    private fun merge(left: List<Int>, right: List<Int>): List<Int> {
        val m = left.size
        val n = right.size
        var i = 0
        var j = 0
        return buildList {
            while (i < m || j < n) {
                if (i >= m) {
                    add(right[j++])
                    continue
                }
                if (j >= n) {
                    add(left[i++])
                    continue
                }
                val a = left[i]
                val b = right[j]
                if (a <= b) {
                    add(a)
                    i++
                } else {
                    add(b)
                    j++
                }
            }
        }

    }

    fun getSmallerCount(k: Int, i: Int, j: Int): Int {
        return query(1, 0, n - 1, k, i, j)
    }

    private fun query(node: Int, l: Int, r: Int, k: Int, i: Int, j: Int): Int {
        if (r < i || l > j) return 0

        if (i <= l && r <= j) {
            val idx = countLessThan(tree[node], k)
            return if (idx >= 0) idx else -idx - 1
        }

        val mid = (l + r) / 2
        val left = query(node * 2, l, mid, k, i, j)
        val right = query(node * 2 + 1, mid + 1, r, k, i, j)
        return left + right
    }

    private fun countLessThan(list: List<Int>, k: Int): Int {
        var l = 0
        var r = list.size
        while (l < r) {
            val mid = (l + r) / 2
            if (list[mid] < k) l = mid + 1 else r = mid
        }
        return l
    }
}

class RightFloatSmallerSegmentTree(private val data: DoubleArray) {
    private val n = data.size
    private val tree = Array(4 * n) { emptyList<Double>() }

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = listOf(data[l])
        } else {
            val mid = (l + r) / 2
            build(node * 2, l, mid)
            build(node * 2 + 1, mid + 1, r)
            val left = tree[node * 2]
            val right = tree[node * 2 + 1]
            tree[node] = merge(left, right)
        }
    }

    private fun merge(left: List<Double>, right: List<Double>): List<Double> {
        val m = left.size
        val n = right.size
        var i = 0
        var j = 0
        return buildList {
            while (i < m || j < n) {
                if (i >= m) {
                    add(right[j++])
                    continue
                }
                if (j >= n) {
                    add(left[i++])
                    continue
                }
                val a = left[i]
                val b = right[j]
                if (a <= b) {
                    add(a)
                    i++
                } else {
                    add(b)
                    j++
                }
            }
        }

    }

    fun getSmallerCount(value: Double, i: Int, j: Int): Int {
        return query(1, 0, n - 1, value, i, j)
    }

    private fun query(node: Int, l: Int, r: Int, value: Double, i: Int, j: Int): Int {
        if (r < i || l > j) return 0

        if (i <= l && r <= j) {
            val idx = countLessThan(tree[node], value)
            return if (idx >= 0) idx else -idx - 1
        }

        val mid = (l + r) / 2
        val left = query(node * 2, l, mid, value, i, j)
        val right = query(node * 2 + 1, mid + 1, r, value, i, j)
        return left + right
    }

    private fun countLessThan(list: List<Double>, value: Double): Int {
        var l = 0
        var r = list.size
        while (l < r) {
            val mid = (l + r) / 2
            if (list[mid] < value) l = mid + 1 else r = mid
        }
        return l
    }
}