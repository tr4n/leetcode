import java.util.*

class FurthestBuildingSegmentTree(private val data: List<Int>, private val limit: Int) {
    private val n = data.size
    private val tree = Array(4 * n) { Node() }

    class Node(
        var list: List<Int> = emptyList(),
        var topSum: Long = 0L,
        var totalSum: Long = 0L,
    )

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = Node(
                list = listOf(data[l]),
                topSum = data[l].toLong(),
                totalSum = data[l].toLong(),
            )
            return
        }
        val mid = (l + r) / 2
        build(node * 2, l, mid)
        build(node * 2 + 1, mid + 1, r)
        val left = tree[node * 2]
        val right = tree[node * 2 + 1]
        tree[node] = merge(left, right)
    }

    private fun merge(left: Node, right: Node): Node {
        val pq = PriorityQueue<Int>() // min-heap
        for (v in left.list) {
            pq.add(v)
            if (pq.size > limit) pq.poll()
        }
        for (v in right.list) {
            pq.add(v)
            if (pq.size > limit) pq.poll()
        }

        val topSum = pq.sumOf { it.toLong() }
        return Node(pq.toList(), topSum, left.totalSum + right.totalSum)
    }

    fun query(start: Int, end: Int): Pair<Long, Long> {
        val treeNode = query(1, 0, n - 1, start, end)
        //    println("${treeNode.list}, ${treeNode.topSum}, ${treeNode.totalSum}")
        return Pair(treeNode.topSum, treeNode.totalSum)
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int): Node {
        if (r < ql || l > qr) return Node()

        if (ql <= l && r <= qr) {
            val treeNode = tree[node]
            return treeNode
            //Pair(treeNode.topSum, treeNode.totalSum)
        }

        val mid = (l + r) / 2
        val left = query(node * 2, l, mid, ql, qr)
        val right = query(node * 2 + 1, mid + 1, r, ql, qr)
        return merge(left, right)
    }
}

fun furthestBuilding(heights: IntArray, bricks: Int, ladders: Int): Int {
    val n = heights.size
    val priorityQueue = PriorityQueue<Int>()
    var ladderClimbs = 0L
    var totalClimbs = 0L
    for (i in 1 until n) {
        if (heights[i] <= heights[i - 1]) continue
        val climb = heights[i] - heights[i - 1]
        priorityQueue.add(climb)
        ladderClimbs += climb
        totalClimbs += climb
        if (priorityQueue.size > ladders) {
            ladderClimbs -= priorityQueue.poll()
        }
        val remainingClimbs = totalClimbs - ladderClimbs
        if (remainingClimbs > bricks) {
            return i - 1
        }
    }

    return n - 1
}

fun main() {
    println(
        furthestBuilding(intArrayOf(4, 2, 7, 6, 9, 14, 12), 5, 1)
    )
}