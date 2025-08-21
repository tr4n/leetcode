package local

class RightSumSegmentTree(private val data: LongArray) {
    private val n = data.size
    private val tree = Array(4 * n) { emptyList<Long>() }

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

    private fun merge(left: List<Long>, right: List<Long>): List<Long> {
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

    fun countInRange(lower: Long, upper: Long, start: Int, end: Int): Int {
        return query(1, 0, n - 1, lower, upper, start, end)
    }

    private fun query(node: Int, l: Int, r: Int, lower: Long, upper: Long, start: Int, end: Int): Int {
        if (r < start || l > end) return 0

        if (start <= l && r <= end) {
            val lowerBound = lowerBoundInRange(tree[node], lower, 0, tree[node].size - 1)
            if (lowerBound < 0) return 0
            val upperBound = upperBoundInRange(tree[node], upper, 0, tree[node].size - 1)
            if (upperBound < 0) return 0
            return upperBound - lowerBound + 1
        }

        val mid = (l + r) / 2
        val left = query(node * 2, l, mid, lower, upper, start, end)
        val right = query(node * 2 + 1, mid + 1, r, lower, upper, start, end)
        return left + right
    }


    private fun lowerBoundInRange(arr: List<Long>, lower: Long, l: Int, r: Int): Int {
        var left = l
        var right = r
        var res = -1
        while (left <= right) {
            val mid = left + (right - left) / 2
            if (arr[mid] >= lower) {
                res = mid
                right = mid - 1
            } else {
                left = mid + 1
            }
        }
        return res
    }

    private fun upperBoundInRange(arr: List<Long>, upper: Long, l: Int, r: Int): Int {
        var left = l
        var right = r
        var res = -1
        while (left <= right) {
            val mid = left + (right - left) / 2
            if (arr[mid] <= upper) {
                res = mid
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
        return res
    }
}