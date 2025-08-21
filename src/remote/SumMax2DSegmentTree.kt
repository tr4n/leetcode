package remote

class MaxLongSparseTable(private val arr: List<Long>) {
    private val n = arr.size
    private val log = IntArray(n + 1)
    private val stMax: Array<LongArray>

    init {
        for (i in 2..n) log[i] = log[i / 2] + 1

        val k = log[n] + 1
        stMax = Array(n) { LongArray(k) }

        for (i in 0 until n) {
            stMax[i][0] = arr[i]
        }

        for (k in 1 until k) {
            var i = 0
            while (i + (1 shl k) <= n) {
                stMax[i][k] = minOf(stMax[i][k - 1], stMax[i + (1 shl (k - 1))][k - 1])
                i++
            }
        }
    }

    fun queryMax(start: Int, end: Int): Long {
        val len = end - start + 1
        val k = log[len]
        return maxOf(stMax[start][k], stMax[end - (1 shl k) + 1][k])
    }
}


class SumMax2DSegmentTree(val arr1: IntArray, val arr2: IntArray) {
    data class Node(
        val pairs: List<Pair<Int, Int>> = emptyList(),
        val secondList: List<Int> = emptyList<Int>(),
        val tree: MaxLongSparseTable = MaxLongSparseTable(emptyList())
    )

    private val n = arr1.size
    private val tree = Array(4 * n) { Node() }

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            val pairs = listOf(arr1[l] to arr2[l])
            val secondList = listOf(arr2[l])
            val sTree = MaxLongSparseTable(listOf(arr1[l].toLong() + arr2[l].toLong()))
            tree[node] = Node(pairs, secondList, sTree)
            return
        }
        val mid = (l + r) / 2
        build(2 * node, l, mid)
        build(2 * node + 1, mid + 1, r)
        val left = tree[2 * node]
        val right = tree[2 * node + 1]
        tree[node] = merge(left, right)
    }

    private fun merge(left: Node, right: Node): Node {
        val pairs = mutableListOf<Pair<Int, Int>>()
        val a = left.pairs
        val b = right.pairs
        var i = 0
        var j = 0
        while (i < a.size && j < b.size) {
            if (a[i].first <= b[j].first) {
                pairs.add(a[i++])
            } else {
                pairs.add(b[j++])
            }
        }
        while (i < a.size) pairs.add(a[i++])
        while (j < b.size) pairs.add(b[j++])

        val secondList = pairs.map { it.second }
        val sumList = pairs.map { it.first.toLong() + it.second.toLong() }
        val tree = MaxLongSparseTable(sumList)
        return Node(pairs, secondList, tree)
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int, x: Int, y: Int): Long {
        if (l > qr || r < ql) return -1
        if (ql <= l && r <= qr) {
            return calculateMax(tree[node], x, y)
        }
        val mid = (l + r) / 2
        val left = query(2 * node, l, mid, ql, qr, x, y)
        val right = query(2 * node + 1, mid + 1, r, ql, qr, x, y)
        return maxOf(left, right)
    }

    private fun calculateMax(
        node: Node,
        x: Int,
        y: Int
    ): Long {
        val (list, secondList, sTree) = node
        val fromIndex = findFirstGreaterIndex(list, 0, x) { it.first }
        if (fromIndex == -1) return -1
        val fromIndex2 = findFirstGreaterIndex(secondList, fromIndex, y) { it }
        if (fromIndex2 == -1) return -1
        return sTree.queryMax(fromIndex2, list.size - 1)
    }

    fun getMaxSum(start: Int, end: Int, x: Int, y: Int): Long {
        return query(1, 0, n - 1, start, end, x, y)
    }

    private fun <T> findFirstGreaterIndex(list: List<T>, fromIndex: Int, k: Int, value: (T) -> Int): Int {
        if (list.isEmpty()) return -1
        var l = fromIndex
        var r = list.size - 1
        var result = -1
        while (l <= r) {
            val mid = (l + r) / 2
            if (value(list[mid]) > k) {
                result = mid
                r = mid - 1
            } else l = mid + 1
        }
        return result
    }

    private fun <T> findFirstSmallerIndex(list: List<T>, fromIndex: Int, k: Int, value: (T) -> Int): Int {
        var l = fromIndex
        var r = list.size - 1
        var result = -1
        while (l <= r) {
            val mid = (l + r) / 2
            if (value(list[mid]) < k) {
                result = mid
                l = mid + 1
            } else r = mid - 1
        }
        return result
    }
}

fun maximumSumQueries(nums1: IntArray, nums2: IntArray, queries: Array<IntArray>): IntArray {
    val tree = SumMax2DSegmentTree(nums1, nums2)
    val n = nums1.size

    return IntArray(queries.size) {
        val (x, y) = queries[it]
        tree.getMaxSum(0, n - 1, x, y).toInt()
    }
}