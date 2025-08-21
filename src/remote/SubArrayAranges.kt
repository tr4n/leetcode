package remote

import java.util.Stack

class SubArrayRanges(private val data: IntArray) {
    private val n = data.size
    private val tree = Array(4 * n) { Node(0, 0) }

    class Node(val min: Int, val max: Int)

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = Node(data[l], data[r])
            return
        }
        val mid = (l + r) / 2
        build(2 * node, l, mid)
        build(2 * node + 1, mid + 1, r)
        val leftNode = tree[2 * node]
        val rightNode = tree[2 * node + 1]
        tree[node] = Node(
            min = minOf(leftNode.min, rightNode.min),
            max = maxOf(leftNode.max, rightNode.max),
        )
    }

    fun query(start: Int, end: Int): Long {
        val node = query(1, 0, n - 1, start, end)
        return node.max.toLong() - node.min
    }


    private fun query(node: Int, l: Int, r: Int, ul: Int, ur: Int): Node {
        if (l > ur || r < ul) return Node(Int.MAX_VALUE, Int.MIN_VALUE)

        if (ul <= l && r <= ur) {
            return tree[node]
        }
        val mid = (l + r) / 2
        val left = query(2 * node, l, mid, ul, ur)
        val right = query(2 * node + 1, mid + 1, r, ul, ur)
        return Node(
            min = minOf(left.min, right.min),
            max = maxOf(left.max, right.max),
        )
    }
}

fun subArrayRanges(nums: IntArray): Long {
    val module = 1_000_000_007
    val n = nums.size
    val stack = Stack<Int>()

    var sum = 0L
    for(i in 0 .. n) {
        val current = if(i < n) nums[i] else Int.MAX_VALUE

        while(stack.isNotEmpty() && nums[stack.peek()] > current) {
            val mid = stack.removeLast()
            val start = if(stack.isEmpty()) -1 else stack.peek()
            val end = i
            val count = (mid - start) * (mid - end)
            sum += (count * nums[mid] % module)
        }

        stack.push(i)
    }
    return sum % module

}

fun main() {
    println(
        subArrayRanges(intArrayOf(1, 2, 3))
    )
}