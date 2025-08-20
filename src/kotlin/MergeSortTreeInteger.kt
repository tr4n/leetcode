package org.example

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

fun main(){
    println(
        numberOfPairs(
            intArrayOf(3,2,5),
            intArrayOf(2,2,1),
            1
        )
    )
}