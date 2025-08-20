package org.example

import java.util.*

class WizardStrength(private val data: IntArray, val module: Int) {
    private val n = data.size
    private val tree = Array(4 * n) { Node(0, 0, 0) }

    class Node(
        val sum: Long,
        val prefix: Long,
        val suffix: Long,
    )

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            val value = data[l].toLong()
            tree[node] = Node(
                sum = value,
                prefix = value,
                suffix = value,
            )
            return
        }

        val mid = (l + r) / 2
        build(2 * node, l, mid)
        build(2 * node + 1, mid + 1, r)
        val leftNode = tree[2 * node]
        val rightNode = tree[2 * node + 1]
        val leftSize = (mid - l + 1).toLong()
        val rightSize = (r - mid).toLong()
        tree[node] = merge(leftNode, rightNode, leftSize, rightSize)
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int): Node {
        if (l > qr || r < ql) return Node(0, 0, 0)

        if (ql <= l && r <= qr) {
            return tree[node]
        }

        val mid = (l + r) / 2
        val leftNode = query(2 * node, l, mid, ql, qr)
        val rightNode = query(2 * node + 1, mid + 1, r, ql, qr)
        val leftSize = (minOf(mid, qr) - maxOf(l, ql) + 1).toLong().coerceAtLeast(0)
        val rightSize = (minOf(r, qr) - maxOf(mid + 1, ql) + 1).toLong().coerceAtLeast(0)
        return merge(leftNode, rightNode, leftSize, rightSize)
    }

    private fun merge(leftNode: Node, rightNode: Node, leftSize: Long, rightSize: Long): Node {
        return Node(
            sum = sumModule(leftNode.sum, rightNode.sum),
            prefix = sumModule(leftNode.prefix, rightNode.prefix, leftNode.sum * rightSize),
            suffix = sumModule(leftNode.suffix, rightNode.suffix, rightNode.sum * leftSize),
        )
    }

    private fun sumModule(vararg elements: Long): Long {
        var res = 0L
        for (x in elements) {
            res = (res + x % module) % module
        }
        return res
    }

    fun query(start: Int, end: Int): Pair<Long, Long> {
        val node = query(1, 0, n - 1, start, end)
        //   println("${start}-${end}: ${data.toList().subList(start, end + 1)}: ${node.prefix} ${node.suffix}")
        return node.prefix to node.suffix
    }
}


fun totalStrength(strength: IntArray): Int {
    val module = 1_000_000_007
    val n = strength.size
 //   println(strength.toList())

    val segmentTree = WizardStrength(strength, module)

    val stack = Stack<Int>()

    var sum = 0L
    for (i in 0..n) {
        val current = if (i < n) strength[i] else Int.MIN_VALUE

        while (stack.isNotEmpty() && strength[stack.peek()] > current) {
            val mid = stack.removeLast()
            val start = 1 + if (stack.isEmpty()) -1 else stack.peek()
            val end = i - 1
         //   println("${strength.toList().subList(start, end + 1)} ${strength[mid]}")
            val leftLen = mid - start + 1
            val rightLen = end - mid + 1
            val (prefixSum, _) = segmentTree.query(mid, end)
            val (_, suffixSum) = segmentTree.query(start, mid)

            val min = strength[mid].toLong()
            val totalSum = (prefixSum * leftLen + suffixSum * rightLen - min * leftLen * rightLen) % module
            sum += (totalSum * min) % module
        }

        stack.push(i)
    }

    return (sum % module).toInt()
}

fun main() {
    println(
        totalStrength(intArrayOf(1, 3, 1, 2))
    )
}