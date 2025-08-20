package org.example

class SegmentTreeMinMax(private val arr: IntArray) {

    private val n = arr.size
    private val tree = Array(4 * n) { Pair(Int.MAX_VALUE, Int.MIN_VALUE) }

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = Pair(arr[l], arr[l])
        } else {
            val mid = (l + r) / 2
            build(node * 2, l, mid)
            build(node * 2 + 1, mid + 1, r)
            tree[node] = merge(tree[node * 2], tree[node * 2 + 1])
        }
    }

    private fun merge(a: Pair<Int, Int>, b: Pair<Int, Int>) =
        Pair(minOf(a.first, b.first), maxOf(a.second, b.second))

    fun query(l: Int, r: Int, node: Int = 1, nl: Int = 0, nr: Int = n - 1): Pair<Int, Int> {
        if (r < nl || nr < l) return Pair(Int.MAX_VALUE, Int.MIN_VALUE)
        if (l <= nl && nr <= r) return tree[node]
        val mid = (nl + nr) / 2
        val left = query(l, r, node * 2, nl, mid)
        val right = query(l, r, node * 2 + 1, mid + 1, nr)
        return merge(left, right)
    }

    fun update(idx: Int, value: Int, node: Int = 1, l: Int = 0, r: Int = n - 1) {
        if (l == r) {
            tree[node] = Pair(value, value)
        } else {
            val mid = (l + r) / 2
            if (idx <= mid) update(idx, value, node * 2, l, mid)
            else update(idx, value, node * 2 + 1, mid + 1, r)
            tree[node] = merge(tree[node * 2], tree[node * 2 + 1])
        }
    }
}

class MinSegmentTree(private val arr: IntArray) {

    private val n = arr.size
    private val tree = Array(4 * n) { Int.MAX_VALUE }

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = arr[l]
        } else {
            val mid = (l + r) / 2
            build(node * 2, l, mid)
            build(node * 2 + 1, mid + 1, r)
            tree[node] = minOf(tree[node * 2], tree[node * 2 + 1])
        }
    }

    fun query(l: Int, r: Int, node: Int = 1, nl: Int = 0, nr: Int = n - 1): Int {
        if (r < nl || nr < l) return Int.MAX_VALUE
        if (l <= nl && nr <= r) return tree[node]
        val mid = (nl + nr) / 2
        val left = query(l, r, node * 2, nl, mid)
        val right = query(l, r, node * 2 + 1, mid + 1, nr)
        return minOf(left, right)
    }

    fun update(idx: Int, value: Int, node: Int = 1, l: Int = 0, r: Int = n - 1) {
        if (l == r) {
            tree[node] = value
        } else {
            val mid = (l + r) / 2
            if (idx <= mid) update(idx, value, node * 2, l, mid)
            else update(idx, value, node * 2 + 1, mid + 1, r)
            tree[node] = minOf(tree[node * 2], tree[node * 2 + 1])
        }
    }
}

class MaxSegmentTree(private val arr: IntArray) {

    private val n = arr.size
    private val tree = Array(4 * n) { Int.MIN_VALUE }

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = arr[l]
        } else {
            val mid = (l + r) / 2
            build(node * 2, l, mid)
            build(node * 2 + 1, mid + 1, r)
            tree[node] = maxOf(tree[node * 2], tree[node * 2 + 1])
        }
    }

    private fun query(l: Int, r: Int, node: Int = 1, nl: Int = 0, nr: Int = n - 1): Int {
        if (r < nl || nr < l) return Int.MIN_VALUE
        if (l <= nl && nr <= r) return tree[node]
        val mid = (nl + nr) / 2
        val left = query(l, r, node * 2, nl, mid)
        val right = query(l, r, node * 2 + 1, mid + 1, nr)
        return maxOf(left, right)
    }

    fun getMax(start: Int, end: Int): Int {
        return query(start, end)
    }

    fun update(idx: Int, value: Int) {
        update(idx, value, 1, 0, n - 1)
    }

    private fun update(idx: Int, value: Int, node: Int = 1, l: Int = 0, r: Int = n - 1) {
        if (l == r) {
            tree[node] = value
        } else {
            val mid = (l + r) / 2
            if (idx <= mid) update(idx, value, node * 2, l, mid)
            else update(idx, value, node * 2 + 1, mid + 1, r)
            tree[node] = maxOf(tree[node * 2], tree[node * 2 + 1])
        }
    }
}

class MaxDynamicSegmentTreeInteger(private val L: Int = 0, private val R: Int = 1_000_000_000) {
    private class Node {
        var value = 0
        var left: Node? = null
        var right: Node? = null
    }

    private val root = Node()

    // Point update: set arr[idx] = max(arr[idx], v)
    fun update(idx: Int, v: Int) {
        update(root, L, R, idx, v)
    }

    private fun update(node: Node, l: Int, r: Int, idx: Int, v: Int) {
        if (l == r) {
            node.value = maxOf(node.value, v)
            return
        }
        val mid = (l + r) ushr 1
        if (idx <= mid) {
            if (node.left == null) node.left = Node()
            update(node.left!!, l, mid, idx, v)
        } else {
            if (node.right == null) node.right = Node()
            update(node.right!!, mid + 1, r, idx, v)
        }
        val leftVal = node.left?.value ?: 0
        val rightVal = node.right?.value ?: 0
        node.value = maxOf(leftVal, rightVal)
    }

    // Range max query
    fun query(qL: Int, qR: Int): Int {
        return query(root, L, R, qL, qR)
    }

    private fun query(node: Node?, l: Int, r: Int, qL: Int, qR: Int): Int {
        if (node == null || qR < l || r < qL) return 0
        if (qL <= l && r <= qR) return node.value
        val mid = (l + r) ushr 1
        val leftVal = query(node.left, l, mid, qL, qR)
        val rightVal = query(node.right, mid + 1, r, qL, qR)
        return maxOf(leftVal, rightVal)
    }
}

class MaxDynamicSegmentTreeLong(private val L: Long, private val R: Long) {
    private class Node {
        var value = Long.MIN_VALUE
        var left: Node? = null
        var right: Node? = null
    }

    private val root = Node()

    fun update(idx: Long, v: Long) {
        update(root, L, R, idx, v)
    }

    private fun update(node: Node, l: Long, r: Long, idx: Long, v: Long) {
        if (l == r) {
            node.value = maxOf(node.value, v)
            return
        }
        val mid = l + (r - l) / 2
        if (idx <= mid) {
            if (node.left == null) node.left = Node()
            update(node.left!!, l, mid, idx, v)
        } else {
            if (node.right == null) node.right = Node()
            update(node.right!!, mid + 1, r, idx, v)
        }
        val leftVal = node.left?.value ?: 0L
        val rightVal = node.right?.value ?: 0L
        node.value = maxOf(leftVal, rightVal)
    }

    // Range max query
    fun query(qL: Long, qR: Long): Long {
        return query(root, L, R, qL, qR)
    }

    private fun query(node: Node?, l: Long, r: Long, qL: Long, qR: Long): Long {
        if (node == null || qR < l || r < qL) return 0L
        if (qL <= l && r <= qR) return node.value
        val mid = l + (r - l) / 2
        val leftVal = query(node.left, l, mid, qL, qR)
        val rightVal = query(node.right, mid + 1, r, qL, qR)
        return maxOf(leftVal, rightVal)
    }
}

