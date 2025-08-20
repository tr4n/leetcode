package org.example

class LUPrefix(size: Int) {
    data class Node(
        val l: Int = 0,
        val r: Int = 0,
        val maxSum: Int = 0,
        val prefix: Int = 0,
        val suffix: Int = 0
    ) {
        fun isCovered() = maxSum > 0 && (r - l + 1) == maxSum
    }

    private val n = size + 1
    private val tree = Array(4 * n) { Node() }

    init {
        build(1, 0, n-1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            //  println("$l $r : ${tree[node]}")
            val value = 0
            tree[node] = Node(l, l, value, value, value)
            return
        }
        val mid = (l + r) / 2
        build(2 * node, l, mid)
        build(2 * node + 1, mid + 1, r)
        tree[node] = merge(tree[2 * node], tree[2 * node + 1])
    }

    private fun merge(left: Node, right: Node): Node {
        return Node(
            l = left.l,
            r = right.r,
            maxSum = maxOf(left.maxSum, right.maxSum, left.suffix + right.suffix),
            prefix = if (left.isCovered()) left.maxSum + right.prefix else left.prefix,
            suffix = if (right.isCovered()) right.maxSum + left.suffix else right.suffix
        )
    }


    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int): Node? {
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


    private fun update(node: Int, l: Int, r: Int, value: Int) {
        if (l == r) {
            tree[node] = Node(l, l, 1, 1, 1)
            return
        }
        val mid = (l + r) / 2
        if (value <= mid) update(2 * node, l, mid, value)
        else update(2 * node + 1, mid + 1, r, value)
        tree[node] = merge(tree[2 * node], tree[2 * node + 1])
    }

    fun upload(video: Int) {
        update(1, 0, n - 1, video - 1)
    }

    fun longest(): Int {
        val node = query(1, 0, n - 1, 0, n - 1)
      //  println(node)
        return node?.prefix ?: 0
    }
}

fun main() {
    val server = LUPrefix(4)
    server.upload(3)
    server.longest()
    server.upload(1)
    server.longest();
    server.upload(2)
    server.longest()
}