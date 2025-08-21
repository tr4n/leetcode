package remote

class CountOrSegmentTree(private val data: IntArray) {
    private val n = data.size
    private val tree = Array(4 * n) { Node(0, setOf()) }

    private class Node(val value: Int, val set: Set<Int>)

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = Node(data[l], setOf(data[l]))
            return
        }
        val mid = (l + r) / 2
        build(2 * node, l, mid)
        build(2 * node + 1, mid + 1, r)

        val left = tree[2 * node]
        val right = tree[2 * node + 1]
        val value = left.value or right.value
        tree[node] = Node(
            value = value,
            set = setOf(left.value, right.value, value)
        )
    }

    fun countOR(start: Int, end: Int): Int {
        val set = query(1, 0, n - 1, start, end)
        println(set)
        return set.size
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int): Set<Int> {
        if (l > qr || r < ql) return setOf()

        if (ql <= l && r <= qr) return tree[node].set

        val mid = (l + r) / 2
        val left = query(2 * node, l, mid, ql, qr)
        val right = query(2 * node + 1, mid + 1, r, ql, qr)

        return left + right
    }
}

fun subarrayBitwiseORs(arr: IntArray): Int {
    val resultSet = mutableSetOf<Int>()
    var prev = mutableSetOf<Int>()
    for (num in arr) {
        val next = mutableSetOf<Int>()
        next.add(num)
        for (v in prev) next.add(v or num)
        prev = next
        resultSet.addAll(prev)
    }
    return resultSet.size
}

fun main() {
    println(
        subarrayBitwiseORs(intArrayOf(1,2,4))
    )
}
