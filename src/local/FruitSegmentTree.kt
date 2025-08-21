package local

class FruitSegmentTree {
    private val size = 1_000_000_000
    private val root = Node()

    fun add(number: Int) {
        update(root, 0, size, number)
    }

    fun countInRange(from: Int, to: Int): Int {
        return query(root, 0, size, from, to)
    }

    fun countLessThan(x: Int): Int {
        return query(root, 0, size, 0, x - 1)
    }

    fun countEqual(x: Int): Int {
        return query(root, 0, size, x, x)
    }

    private fun update(node: Node, l: Int, r: Int, pos: Int) {
        node.count++
        if (l == r) return
        val mid = (l + r) / 2
        if (pos <= mid) {
            if (node.left == null) node.left = Node()
            update(node.left!!, l, mid, pos)
        } else {
            if (node.right == null) node.right = Node()
            update(node.right!!, mid + 1, r, pos)
        }
    }

    private fun query(node: Node?, l: Int, r: Int, ql: Int, qr: Int): Int {
        if (node == null || l > qr || r < ql) return 0
        if (ql <= l && r <= qr) return node.count

        val mid = (l + r) / 2
        return query(node.left, l, mid, ql, qr) +
                query(node.right, mid + 1, r, ql, qr)
    }

    private class Node {
        var left: Node? = null
        var right: Node? = null
        var count: Int = 0
    }
}