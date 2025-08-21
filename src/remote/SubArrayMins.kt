package remote

import java.util.*

class SubArrayMins(private val data: IntArray) {
    private val n = data.size
    private val tree = IntArray(4 * n)

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = data[l]
            return
        }
        val mid = (l + r) / 2
        build(2 * node, l, mid)
        build(2 * node + 1, mid + 1, r)
        val leftNode = tree[2 * node]
        val rightNode = tree[2 * node + 1]
        tree[node] = minOf(leftNode, rightNode)
    }

    fun query(start: Int, end: Int): Int {
        val node = query(1, 0, n - 1, start, end)
        return node
    }


    private fun query(node: Int, l: Int, r: Int, ul: Int, ur: Int): Int {
        if (l > ur || r < ul) return Int.MAX_VALUE

        if (ul <= l && r <= ur) {
            return tree[node]
        }
        val mid = (l + r) / 2
        val left = query(2 * node, l, mid, ul, ur)
        val right = query(2 * node + 1, mid + 1, r, ul, ur)
        return minOf(left, right)
    }
}

fun sumSubarrayMins(nums: IntArray): Int {
    val module = 1_000_000_007
    val n = nums.size
    val stack = Stack<Int>()

    var sum = 0L
    for (i in 0..n) {
        val current = if (i < n) nums[i] else Int.MIN_VALUE

        while (stack.isNotEmpty() && nums[stack.peek()] > current) {
            val mid = stack.removeLast()
            val start = if (stack.isEmpty()) -1 else stack.peek()
            val end = i
            val count = (mid - start) * (end - mid).toLong()
            sum += (count * nums[mid] % module)
        }

        stack.push(i)
    }
    return (sum % module).toInt()

}

fun main() {
    println(
        sumSubarrayMins(intArrayOf(3, 1, 2, 4))
    )
}