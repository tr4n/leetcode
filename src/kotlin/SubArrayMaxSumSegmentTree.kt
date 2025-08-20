package org.example

class SubArrayMaxSumSegmentTree(private val arr: IntArray) {

    data class Node(
        val totalSum: Long = -1_000_003L,
        val prefixSum: Long = -1_000_003L,
        val suffixSum: Long = -1_000_003L,
        val maxSum: Long = -1_000_003L,
    )

    private val n = arr.size
    private val tree = Array(4 * n) { Node() }

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            val value = arr[l].toLong()
            tree[node] = Node(value, value, value, value)
        } else {
            val mid = (l + r) / 2
            build(node * 2, l, mid)
            build(node * 2 + 1, mid + 1, r)
            tree[node] = merge(tree[node * 2], tree[node * 2 + 1])
        }
    }

    private fun merge(a: Node, b: Node) = Node(
        totalSum = a.totalSum + b.totalSum,
        prefixSum = maxOf(a.prefixSum, a.totalSum + b.prefixSum),
        suffixSum = maxOf(a.suffixSum + b.totalSum, b.suffixSum),
        maxSum = maxOf(a.maxSum, b.maxSum, a.suffixSum + b.prefixSum)
    )

    fun getSubArrayMaxSum(start: Int, end: Int): Node? {
        return query(1, 0, n - 1, start, end)
    }

    fun queryMaxSum(segments: List<Pair<Int, Int>>): Long {
        val nodes = segments.mapNotNull { (start, end) -> getSubArrayMaxSum(start, end) }
        if (nodes.isEmpty()) return Long.MIN_VALUE
        var currentMax = Long.MIN_VALUE
        var currentNode: Node? = null

        for (node in nodes) {
            currentNode = if (currentNode == null) node else merge(currentNode, node)

            if (currentNode.maxSum > currentMax) {
                currentMax = currentNode.maxSum
            }
        }

        return currentMax
    }

    private fun query(node: Int = 1, l: Int, r: Int, ql: Int = 0, qr: Int = n - 1): Node? {
        if (l > qr || r < ql) return null
        if (ql <= l && r <= qr) return tree[node]
        val mid = (l + r) / 2
        val left = query(2 * node, l, mid, ql, qr)
        val right = query(2 * node + 1, mid + 1, r, ql, qr)
        return when {
            left != null && right != null -> merge(left, right)
            left != null -> left
            right != null -> right
            else -> null
        }
    }
}

fun maximumSum(arr: IntArray): Int {
    val n = arr.size
    val tree = SubArrayMaxSumSegmentTree(arr)

    var maxSum = arr.sumOf { it.toLong() }
    maxSum = maxOf(maxSum, arr.max().toLong())
    for (i in 0 until n) {
        val segments = mutableListOf<Pair<Int, Int>>()
        if (i > 0) {
            segments.add(0 to i - 1)
        }
        if (n > i + 1) {
            segments.add(i + 1 to n - 1)
        }

        maxSum = maxOf(maxSum, tree.queryMaxSum(segments))
    }
    return maxSum.toInt()
}

fun maxSubarraySum(nums: IntArray): Long {
    val n = nums.size
    val tree = SubArrayMaxSumSegmentTree(nums)

    val numToIndexes = mutableMapOf<Int, MutableList<Int>>()
    for (i in 0 until n) {
        val num = nums[i]
        val list = numToIndexes[num]
        if (list.isNullOrEmpty()) {
            numToIndexes[num] = mutableListOf(i)
        } else {
            numToIndexes[num]?.add(i)
        }
    }


    var maxSum = nums.sumOf { it.toLong() }
    maxSum = maxOf(maxSum, nums.max().toLong())
    for (indexes in numToIndexes.values) {
        val segments = mutableListOf<Pair<Int, Int>>()
        var start = 0
        for (i in indexes) {
            if (i > start) {
                segments.add(start to i - 1)
            }
            start = i + 1
        }
        if (n > start) {
            segments.add(start to n - 1)
        }

        maxSum = maxOf(maxSum, tree.queryMaxSum(segments))
    }
    //   println(nums.toList())
    return maxSum
}

fun main() {
    println(
        maxSubarraySum(intArrayOf(-31, -23, -47))
    )
}