package org.example

class CountIntervals {
    private val root = Node(0, 1_000_000_000)

    fun add(left: Int, right: Int) {
        update(root, left, right)
    }

    fun count(): Int {
        return root.value
    }

    private fun query(node: Node?, ql: Int, qr: Int): Int {
        if (node == null || node.l > qr || node.r < ql) {
            return 0
        }

        if (ql <= node.l && node.r <= qr) {
            return node.value
        }

        val mid = (node.l + node.r) / 2
        val left = node.left ?: Node(node.l, mid).also { node.left = it }
        val right = node.right ?: Node(mid + 1, node.r).also { node.right = it }

        val leftCount = query(left, ql, qr)
        val rightCount = query(right, ql, qr)
        return leftCount + rightCount
    }

    private fun update(node: Node?, ul: Int, ur: Int) {
        if (node == null || ul > node.r || ur < node.l) {
            return
        }

        if (node.value == node.r - node.l + 1) return

        if (ul <= node.l && node.r <= ur) {
            node.value = node.r - node.l + 1
            node.left = null
            node.right = null
            return
        }


        val mid = (node.l + node.r) / 2
        if (node.left == null) node.left = Node(node.l, mid)
        if (node.right == null) node.right = Node(mid + 1, node.r)

        update(node.left, ul, ur)
        update(node.right, ul, ur)
        node.value = (node.left?.value ?: 0) + (node.right?.value ?: 0)
    }

    private class Node(val l: Int, val r: Int) {
        var left: Node? = null
        var right: Node? = null
        var value = 0
    }
}

fun main() {
    val tree = CountIntervals()
    tree.add(2, 3)
    tree.add(7, 10)
    println(tree.count())
    tree.add(5, 8)
    println(tree.count())

}