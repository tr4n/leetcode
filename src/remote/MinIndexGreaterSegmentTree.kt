package remote

class LeftMostIndexSegmentTree(private val data: IntArray) {
    private val n = data.size
    private val tree = IntArray(4 * n)

    init {
        build(1, 0, n - 1)
    }


    fun findIndex(value: Int, start: Int, end: Int): Int {
        return query(1, 0, n - 1, start, end, value)
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

        if (l == r) return if (data[index] > value) index else -1

        val mid = (l + r) / 2
        val leftResult = query(2 * node, l, mid, ql, qr, value)
        return if (leftResult != -1) leftResult
        else query(2 * node + 1, mid + 1, r, ql, qr, value)
    }
}

fun leftmostBuildingQueries(heights: IntArray, queries: Array<IntArray>): IntArray {
    val n = queries.size
    val result = IntArray(n)
    val segmentTree = LeftMostIndexSegmentTree(heights)

    for (i in queries.indices) {
        val a = queries[i].min()
        val b = queries[i].max()

        if(a == b) {
            result[i] = a
            continue
        }

        if(heights[a] < heights[b]) {
            result[i] = b
            continue
        }

        val floorHeight = maxOf(heights[a], heights[b])
        result[i] = segmentTree.findIndex(floorHeight, b, heights.size - 1)
    }
    return result
}

fun main(){
    val heights = intArrayOf(1,2,1,2,1,2)
    val queries = arrayOf(
        intArrayOf(0, 0), intArrayOf(0, 1), intArrayOf(0, 2), intArrayOf(0, 3), intArrayOf(0, 4), intArrayOf(0, 5),
        intArrayOf(1, 0), intArrayOf(1, 1), intArrayOf(1, 2), intArrayOf(1, 3), intArrayOf(1, 4), intArrayOf(1, 5),
        intArrayOf(2, 0), intArrayOf(2, 1), intArrayOf(2, 2), intArrayOf(2, 3), intArrayOf(2, 4), intArrayOf(2, 5),
        intArrayOf(3, 0), intArrayOf(3, 1), intArrayOf(3, 2), intArrayOf(3, 3), intArrayOf(3, 4), intArrayOf(3, 5),
        intArrayOf(4, 0), intArrayOf(4, 1), intArrayOf(4, 2), intArrayOf(4, 3), intArrayOf(4, 4), intArrayOf(4, 5),
        intArrayOf(5, 0), intArrayOf(5, 1), intArrayOf(5, 2), intArrayOf(5, 3), intArrayOf(5, 4), intArrayOf(5, 5)
    )
    println(leftmostBuildingQueries(heights, queries).toList())
}