class MinIndexSegmentTree(arr: IntArray) {
    private var data = arr
    private val n = data.size
    private val tree = IntArray(4 * n)

    init {
        build(1, 0, n - 1)
    }

    fun update(index: Int, value: Int) {
        data[index] = value
        update(1, 0, n - 1, index, 0, n - 1)
    }

    fun findIndex(value: Int): Int {
        return query(1, 0, n - 1, 0, n - 1, value)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = l
            return
        }

        val mid = (l + r) / 2
        build(2 * node, l, mid)
        build(2 * node + 1, mid + 1, r)

        val leftNode = tree[2 * node]
        val rightNode = tree[2 * node + 1]
        tree[node] = when {
            data[leftNode] > data[rightNode] -> leftNode
            data[leftNode] < data[rightNode] -> rightNode
            else -> minOf(leftNode, rightNode)
        }
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int, value: Int): Int {
        if (r < ql || l > qr) return -1

        val index = tree[node]
        if (ql <= l && r <= qr && data[index] < value) return -1

        if (l == r) return if (data[index] >= value) index else -1

        val mid = (l + r) / 2
        val leftResult = query(2 * node, l, mid, ql, qr, value)
        return if (leftResult != -1) leftResult
        else query(2 * node + 1, mid + 1, r, ql, qr, value)
    }

    private fun update(node: Int, l: Int, r: Int, index: Int, ul: Int, ur: Int) {
        if (l > ur || r < ul) return

        if (l == r) {
            tree[node] = index
            return
        }

        val mid = (l + r) / 2
        if (index <= mid) update(2 * node, l, mid, index, ul, ur)
        else update(2 * node + 1, mid + 1, r, index, ul, ur)

        val leftNode = tree[2 * node]
        val rightNode = tree[2 * node + 1]
        tree[node] = when {
            data[leftNode] > data[rightNode] -> leftNode
            data[leftNode] < data[rightNode] -> rightNode
            else -> minOf(leftNode, rightNode)
        }
    }

    fun rebuild(arr: IntArray) {
        data = arr
        build(1, 0, n - 1)
    }
}

fun numOfUnplacedFruits(fruits: IntArray, baskets: IntArray): Int {
    val treeMap = MinIndexSegmentTree(baskets)

    var remaining = 0
    for (fruit in fruits) {

        val idx = treeMap.findIndex(fruit)
        if (idx < 0) {
            remaining++
            continue
        }
        treeMap.update(idx, -1)
    }
    return remaining
}

fun main() {
    println(numOfUnplacedFruits(intArrayOf(35, 61), intArrayOf(76, 56)))
}