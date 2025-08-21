package remote

import java.util.*

class Pattern132(private val data: IntArray) {
    private val n = data.size
    private val tree = IntArray(4 * n)

    init {
        build(1, 0, n - 1)
    }


    fun findIndex(value: Int, start: Int, end: Int): Int {
        return query(1, 0, n - 1, start, end, value)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = l
            return
        }

        val mid = (l + r) / 2
        build(2 * node, l, mid)
        build(2 * node + 1, mid + 1, r)

        val leftNode = tree[2 * node]
        val rightNode = tree[2 * node + 1]
        tree[node] = when {
            data[leftNode] > data[rightNode] -> rightNode
            data[leftNode] < data[rightNode] -> leftNode
            else -> minOf(leftNode, rightNode)
        }
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int, value: Int): Int {
        if (r < ql || l > qr) return -1

        val index = tree[node]
        if (ql <= l && r <= qr && data[index] >= value) return -1

        if (l == r) return if (data[index] < value) index else -1

        val mid = (l + r) / 2
        val leftResult = query(2 * node, l, mid, ql, qr, value)
        return if (leftResult != -1) leftResult
        else query(2 * node + 1, mid + 1, r, ql, qr, value)
    }
}

fun find132pattern(nums: IntArray): Boolean {
    val n = nums.size
    if (n < 3) return false
    val tree = Pattern132(nums)

    val stack = Stack<Int>()
    val greaterLeft = IntArray(n)
    for (i in (n - 1) downTo 0) {
        while (stack.isNotEmpty() && nums[i] > nums[stack.peek()]) {
            val end = stack.pop()
            greaterLeft[end] = i
        }
        stack.push(i)
    }

    for (k in (n - 1) downTo 2) {
        val j = greaterLeft[k]
        val i = tree.findIndex(nums[k], 0, j - 1)
     //   println("${nums.getOrNull(i)} ${nums.getOrNull(j)} ${nums.getOrNull(k)}")
        if (i >= 0) {
            return true
        }
    }


    return false
}


fun main() {
    println(
        find132pattern(intArrayOf(-2, 1, 1))
    )
}