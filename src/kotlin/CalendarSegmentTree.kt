package org.example

class CalendarSegmentTree {
    private val root = Node(0, 1_000_000_000)

    fun book(start: Int, end: Int): Int {
        update(root, start, end - 1)
        return query(root, start, end - 1)
    }

    private fun query(node: Node?, ql: Int, qr: Int): Int {
        if (node == null || node.l > qr || node.r < ql) return 0

        if (ql <= node.l && node.r <= qr) return node.value

        val mid = (node.l + node.r) / 2
        val left = node.left ?: Node(node.l, mid).also { node.left = it }
        val right = node.right ?: Node(mid + 1, node.r).also { node.right = it }

        val leftCount = query(left, ql, qr)
        val rightCount = query(right, ql, qr)
        return leftCount + rightCount
    }

    private fun update(node: Node?, ul: Int, ur: Int) {
        if (node == null || ul > node.r || ur < node.l) return

        if (ul <= node.l && node.r <= ur) {
            node.value++
            node.left = null
            node.right = null
            return
        }

        val mid = (node.l + node.r) / 2
        val left = node.left ?: Node(node.l, mid).also { node.left = it }
        val right = node.right ?: Node(mid + 1, node.r).also { node.right = it }
        update(left, ul, ur)
        update(right, ul, ur)
    }

    private class Node(val l: Int, val r: Int) {
        var left: Node? = null
        var right: Node? = null
        var value = 0
    }
}
