package home

class RangeFreqQuery(arr: IntArray) {
    private val data = arr
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

    fun query(left: Int, right: Int, value: Int): Int {
        return query(1, 0, n - 1, value, left, right)
    }


    private fun query(node: Int, l: Int, r: Int, k: Int, i: Int, j: Int): Int {
        if (r < i || l > j) return 0

        if (i <= l && r <= j) {
            val end = findLastIndex(tree[node], k)
            val start = findFirstIndex(tree[node], k)

            return if (start >= 0 && end >= 0) end - start + 1 else 0
        }

        val mid = (l + r) / 2
        val left = query(node * 2, l, mid, k, i, j)
        val right = query(node * 2 + 1, mid + 1, r, k, i, j)
        return left + right
    }

    private fun findFirstIndex(list: List<Int>, k: Int): Int {
        var l = 0
        var r = list.size - 1
        var result = -1
        while (l <= r) {
            val mid = (l + r) / 2
            if (list[mid] < k) {
                l = mid + 1
            } else {
                if (list[mid] == k) result = mid
                r = mid - 1
            }
        }
        return result
    }

    private fun findLastIndex(list: List<Int>, k: Int): Int {
        var l = 0
        var r = list.size - 1
        var result = -1
        while (l <= r) {
            val mid = (l + r) / 2
            if (list[mid] > k) {
                r = mid - 1
            } else {
                if (list[mid] == k) result = mid
                l = mid + 1
            }
        }
        return result
    }

}

fun main() {
    val arr = intArrayOf(12, 33, 4, 56, 22, 2, 34, 33, 22, 12, 34, 56)

    val tree = RangeFreqQuery(arr)
    println(
        tree.query(0, 11, 33)
    )
}