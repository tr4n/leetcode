package remote

class MergeSortTreeInteger(private val arr: IntArray) {
    private val n = arr.size
    private val tree = Array(4 * n) { mutableListOf<Int>() }

    init {
        if (arr.isNotEmpty()) build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = mutableListOf(arr[l])
            return
        }
        val mid = (l + r) / 2
        build(node * 2, l, mid)
        build(node * 2 + 1, mid + 1, r)
        tree[node] = merge(tree[node * 2], tree[node * 2 + 1])
    }

    private fun merge(a: List<Int>, b: List<Int>): MutableList<Int> {
        val result = mutableListOf<Int>()
        var i = 0
        var j = 0
        while (i < a.size && j < b.size) {
            if (a[i] <= b[j]) {
                result.add(a[i++])
            } else {
                result.add(b[j++])
            }
        }
        while (i < a.size) result.add(a[i++])
        while (j < b.size) result.add(b[j++])
        return result
    }

    fun query(ql: Int, qr: Int): List<Int> {
        return query(1, 0, n - 1, ql, qr)
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int): List<Int> {
        if (qr < l || r < ql) return emptyList()
        if (ql <= l && r <= qr) return tree[node]
        val mid = (l + r) / 2
        val left = query(node * 2, l, mid, ql, qr)
        val right = query(node * 2 + 1, mid + 1, r, ql, qr)
        return merge(left, right)
    }

    fun count(ql: Int, qr: Int, onCount: (List<Int>) -> Long): Long {
        return queryCount(1, 0, n - 1, ql, qr, onCount)
    }

    private fun queryCount(node: Int, l: Int, r: Int, ql: Int, qr: Int, onCount: (List<Int>) -> Long): Long {
        if (qr < l || r < ql) return 0
        if (ql <= l && r <= qr) return onCount(tree[node])
        val mid = (l + r) / 2
        val left = queryCount(node * 2, l, mid, ql, qr, onCount)
        val right = queryCount(node * 2 + 1, mid + 1, r, ql, qr, onCount)
        return left + right
    }

    fun update(idx: Int, newValue: Int) {
        update(1, 0, n - 1, idx, newValue)
    }

    private fun update(node: Int, l: Int, r: Int, idx: Int, newValue: Int) {
        val oldValue = arr[idx]
        val list = tree[node]

        val pos = list.binarySearch(oldValue)
        if (pos >= 0) list.removeAt(pos)

        val insertPos = list.binarySearch(newValue).let { if (it < 0) -(it + 1) else it }
        list.add(insertPos, newValue)

        if (l == r) {
            arr[idx] = newValue
            return
        }

        val mid = (l + r) / 2
        if (idx <= mid) {
            update(node * 2, l, mid, idx, newValue)
        } else {
            update(node * 2 + 1, mid + 1, r, idx, newValue)
        }
    }
}


class MergeSortTree<T : Comparable<T>>(private val arr: List<T>) {
    private val n = arr.size
    private val tree = Array(4 * n) { mutableListOf<T>() }

    init {
        if (arr.isNotEmpty()) build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = mutableListOf(arr[l])
            return
        }
        val mid = (l + r) / 2
        build(node * 2, l, mid)
        build(node * 2 + 1, mid + 1, r)
        tree[node] = merge(tree[node * 2], tree[node * 2 + 1])
    }

    private fun merge(a: List<T>, b: List<T>): MutableList<T> {
        val result = mutableListOf<T>()
        var i = 0
        var j = 0
        while (i < a.size && j < b.size) {
            if (a[i] <= b[j]) {
                result.add(a[i++])
            } else {
                result.add(b[j++])
            }
        }
        while (i < a.size) result.add(a[i++])
        while (j < b.size) result.add(b[j++])
        return result
    }

    fun query(ql: Int, qr: Int): List<T> {
        return query(1, 0, n - 1, ql, qr)
    }

    fun count(ql: Int, qr: Int, onCount: (List<T>) -> Long): Long {
        return queryCount(1, 0, n - 1, ql, qr, onCount)
    }

    private fun queryCount(node: Int, l: Int, r: Int, ql: Int, qr: Int, onCount: (List<T>) -> Long): Long {
        if (qr < l || r < ql) return 0
        if (ql <= l && r <= qr) return onCount(tree[node])
        val mid = (l + r) / 2
        val left = queryCount(node * 2, l, mid, ql, qr, onCount)
        val right = queryCount(node * 2 + 1, mid + 1, r, ql, qr, onCount)
        return left + right
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int): List<T> {
        if (qr < l || r < ql) return emptyList()
        if (ql <= l && r <= qr) return tree[node]
        val mid = (l + r) / 2
        val left = query(node * 2, l, mid, ql, qr)
        val right = query(node * 2 + 1, mid + 1, r, ql, qr)
        return merge(left, right)
    }
}

fun numberOfPairs(nums1: IntArray, nums2: IntArray, diff: Int): Long {
    val n = nums1.size
    val nums = (0 until n).map { nums1[it] - nums2[it] }
    val mergeSortTree = MergeSortTree(nums)

    var cnt = 0L
    for (i in 0 until n - 1) {
        val target = nums[i] - diff
        cnt += mergeSortTree.count(i + 1, n - 1) { list ->
            var l = 0
            var r = list.size - 1
            var firstIndex = -1

            while (l <= r) {
                val mid = (l + r) / 2
                val value = list[mid]

                if (value >= target) {
                    firstIndex = mid
                    r = mid - 1
                } else {
                    l = mid + 1
                }
            }
            if (firstIndex < 0) 0L else (list.size - firstIndex).toLong()
        }
    }
    return cnt
}

fun createSortedArray(instructions: IntArray): Int {
    fun countLessThan(list: List<Int>, x: Int): Int {
        var l = 0
        var r = list.size
        while (l < r) {
            val mid = (l + r) / 2
            if (list[mid] < x) l = mid + 1 else r = mid
        }
        return l
    }

    fun countGreaterThan(list: List<Int>, x: Int): Int {
        var l = 0
        var r = list.size
        while (l < r) {
            val mid = (l + r) / 2
            if (list[mid] <= x) l = mid + 1 else r = mid
        }
        return list.size - l
    }

    val mod = 1_000_000_007
    val tree = MergeSortTree(instructions.toList())
    var totalCost = 0L

    for (i in 1 until instructions.size) {
        val num = instructions[i]
        val lessCount = tree.count(0, i - 1) { list ->
            countLessThan(list, num).toLong()
        }
        val greaterCount = tree.count(0, i - 1) { list ->
            countGreaterThan(list, num).toLong()
        }

        val cost = minOf(lessCount, greaterCount)
        totalCost = (totalCost + cost) % mod
    }

    return totalCost.toInt()
}

fun containsNearbyAlmostDuplicate(nums: IntArray, indexDiff: Int, valueDiff: Int): Boolean {
    fun countLE(list: List<Int>, x: Int): Int {
        var l = 0
        var r = list.size
        while (l < r) {
            val mid = (l + r) / 2
            if (list[mid] <= x) l = mid + 1 else r = mid
        }
        return l
    }

    fun countGE(list: List<Int>, x: Int): Int {
        var l = 0
        var r = list.size
        while (l < r) {
            val mid = (l + r) / 2
            if (list[mid] < x) l = mid + 1 else r = mid
        }
        return list.size - l
    }

    val n = nums.size
    val tree = MergeSortTreeInteger(nums)

    for (i in 0 until n - indexDiff) {
        val num = nums[i]
        val start = maxOf(0, i - indexDiff)
        val end = minOf(n - 1, i + indexDiff)
        val countInRange = tree.count(start, end) { list ->
            val le1 = countLE(list, num + valueDiff).toLong()
            val le2 = countLE(list, num - valueDiff - 1).toLong()
            le1 - le2
        }
        if (countInRange > 0L) return true
    }
    return false
}

fun main() {
    println(
        containsNearbyAlmostDuplicate(intArrayOf(1, 5, 9, 1, 5, 9), 2, 3)
    )
}