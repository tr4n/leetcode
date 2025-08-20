package org.example

class PairSegmentTree(n: Int) {
    data class Node(val a: Int, val b: Int, val idx: Int)

    val tree = Array(4 * n) { mutableListOf<Node>() }

    fun build(node: Int, l: Int, r: Int, arr1: IntArray, arr2: IntArray) {
        if (l == r) {
            tree[node] = mutableListOf(Node(arr1[l], arr2[l], l))
        } else {
            val mid = (l + r) / 2
            build(node * 2, l, mid, arr1, arr2)
            build(node * 2 + 1, mid + 1, r, arr1, arr2)
            tree[node] = merge(tree[node * 2], tree[node * 2 + 1])
        }
    }

    // a ↑, b ↓
    fun merge(left: List<Node>, right: List<Node>): MutableList<Node> {
        val res = mutableListOf<Node>()
        var i = 0;
        var j = 0
        while (i < left.size && j < right.size) {
            if (left[i].a < right[j].a ||
                (left[i].a == right[j].a && left[i].b > right[j].b)
            ) {
                res.add(left[i++])
            } else {
                res.add(right[j++])
            }
        }
        while (i < left.size) res.add(left[i++])
        while (j < right.size) res.add(right[j++])
        return res
    }


    fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int, x: Int, y: Int): Int {
        if (qr < l || r < ql) return -1
        if (ql <= l && r <= qr) {
            val arr = tree[node]
            var lo = 0
            var hi = arr.size
            while (lo < hi) {
                val mid = (lo + hi) / 2
                if (arr[mid].a >= x) hi = mid else lo = mid + 1
            }
            for (i in lo until arr.size) {
                if (arr[i].b >= y) {
                    return arr[i].idx
                }
            }
            return -1
        }
        val mid = (l + r) / 2
        val leftAns = query(node * 2, l, mid, ql, qr, x, y)
        val rightAns = query(node * 2 + 1, mid + 1, r, ql, qr, x, y)
        return when {
            leftAns == -1 -> rightAns
            rightAns == -1 -> leftAns
            else -> minOf(leftAns, rightAns)
        }
    }

    fun findIndexWithMaxSum(list: List<Node>): Int {
        val n = list.size
        if (n == 0) return -1
        var l = 0
        var r = n - 1
        while (r - l > 3) {
            val m1 = l + (r - l) / 3
            val m2 = r - (r - l) / 3
            val node1 = list[m1]
            val node2 = list[m2]
            val f1 = node1.a.toLong() + node1.b.toLong()
            val f2 = node2.a.toLong() + node2.b.toLong()
            if (f1 < f2) l = m1 + 1 else r = m2 - 1
        }
        var bestIndex = l
        var bestVal = list[bestIndex].a.toLong() + list[bestIndex].b.toLong()
        for (i in (l + 1)..r) {
            val v = list[i].a.toLong() + list[i].b.toLong()
            if (v > bestVal) {
                bestIndex = i
                bestVal = v
            }
        }
        return bestIndex
    }
}