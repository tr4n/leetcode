package remote

class MergeSort2DSegmentTree(val arr1: IntArray, val arr2: IntArray) {
    data class Node(
        val pairs: List<Pair<Int, Int>> = emptyList(),
        val tree: MergeSortTreeInteger = MergeSortTreeInteger(intArrayOf())
    )

    private val n = arr1.size
    private val tree = Array(4 * n) { Node() }

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            val pairs = listOf(arr1[l] to arr2[l])
            val fenwick = MergeSortTreeInteger(intArrayOf(arr2[l]))
            tree[node] = Node(pairs, fenwick)
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

        val list = pairs.map { it.second }.toIntArray()
        val tree = MergeSortTreeInteger(list)
        return Node(pairs, tree)
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int, x: Int, y: Int): Pair<Int, Int> {
        if (l > qr || r < ql) return 0 to 0
        if (ql <= l && r <= qr) {
            val (list, mst) = tree[node]
            val greater = countGreater(list, mst, x, y)
            val smaller = countSmaller(list, mst, x, y)
            return greater to smaller
        }
        val mid = (l + r) / 2
        val left = query(2 * node, l, mid, ql, qr, x, y)
        val right = query(2 * node + 1, mid + 1, r, ql, qr, x, y)
        return Pair(
            left.first + right.first,
            left.second + right.second
        )
    }

    private fun countGreater(list: List<Pair<Int, Int>>, sTree: MergeSortTreeInteger, x: Int, y: Int): Int {
        val fromIndex = findFirstGreaterIndex(list, 0, x) { it.first }
        if (fromIndex == -1) return 0
        val secondList = sTree.query(fromIndex, list.size - 1)
        val index = findFirstGreaterIndex(secondList, fromIndex, y) { it }
        if (index == -1) return 0
        return secondList.size - index
    }

    private fun countSmaller(list: List<Pair<Int, Int>>, sTree: MergeSortTreeInteger, x: Int, y: Int): Int {
        val fromIndex = findFirstSmallerIndex(list, 0, x) { it.first }
        if (fromIndex == -1) return 0
        val secondList = sTree.query(fromIndex, list.size - 1)
        val index = findFirstSmallerIndex(secondList, fromIndex, y) { it }
        if (index == -1) return 0
        return secondList.size - index
    }

    fun count(start: Int, end: Int, x: Int, y: Int): Pair<Int, Int> {
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

fun goodTriplets(nums1: IntArray, nums2: IntArray): Long {
    val n = nums1.size
    val tree = MergeSort2DSegmentTree(nums1, nums2)
    var cnt = 0L
    for (i in 1 until n - 1) {
        val x = nums1[i]
        val y = nums2[i]
        val smallerLeft = tree.count(0, i - 1, x, y).second.toLong()
        val greaterRight = tree.count(i + 1, n - 1, x, y).first.toLong()
        println("$i $smallerLeft $greaterRight")
        if(smallerLeft == 0L || greaterRight == 0L) continue

        cnt += smallerLeft * greaterRight
    }
    return cnt
}

fun main() {
    println(
        goodTriplets(
            intArrayOf(4,0,1,3,2),
            intArrayOf(4,1,0,2,3)
        )
    )
}