package home

class MaxIndexValueSegmentTree(private val data: LongArray) {
    class Node(val value: Long, val index: Int)

    private val n = data.size
    private val tree = Array(4 * n) { mutableListOf<Node>() }

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = mutableListOf(Node(data[l], l))
            return
        }
        val mid = (l + r) / 2
        build(2 * node, l, mid)
        build(2 * node + 1, mid + 1, r)

        val leftNode = tree[2 * node]
        val rightNode = tree[2 * node + 1]

        tree[node] = merge(leftNode, rightNode)
    }

    private fun merge(a: List<Node>, b: List<Node>): MutableList<Node> {
        val res = mutableListOf<Node>()
        var i = 0
        var j = 0
        var currentMaxIndex = -1
        while (i < a.size && j < b.size) {
            val next = if (a[i].value <= b[j].value) a[i++] else b[j++]
            currentMaxIndex = maxOf(currentMaxIndex, next.index)
            res.add(Node(next.value, currentMaxIndex))
        }
        while (i < a.size) {
            val next = a[i++]
            currentMaxIndex = maxOf(currentMaxIndex, next.index)
            res.add(Node(next.value, currentMaxIndex))
        }
        while (j < b.size) {
            val next = b[j++]
            currentMaxIndex = maxOf(currentMaxIndex, next.index)
            res.add(Node(next.value, currentMaxIndex))
        }
        return res
    }

    fun findMaxIndex(x: Long, from: Int, to: Int): Int {
        return query(1, 0, n - 1, from, to, x)
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int, x: Long): Int {
        if (l > qr || r < ql) return -1
        if (ql <= l && r <= qr) {
            val list = tree[node]
          //  println(list.map { it.value })
            var low = 0
            var high = list.size - 1
            var ans = -1
            while (low <= high) {
                val mid = (low + high) / 2
                if (list[mid].value <= x) {
                    ans = mid
                    low = mid + 1
                } else {
                    high = mid - 1
                }
            }
            return if (ans >= 0) list[ans].index else -1
        }
        val mid = (l + r) / 2
        val left = query(2 * node, l, mid, ql, qr, x)
        val right = query(2 * node + 1, mid + 1, r, ql, qr, x)
        return maxOf(left, right)
    }
}

fun shortestSubarray(nums: IntArray, k: Int): Int {
    val n = nums.size
    if (nums.any { it >= k }) return 1

    var shortestLength = n + 1
    val sums = LongArray(n)
    sums[0] = nums[0].toLong()
    if (sums[0] >= k.toLong()) return 1


    for (i in 1 until n) {
        sums[i] = sums[i - 1] + nums[i].toLong()
        if (sums[i] >= k.toLong()) {
            shortestLength = minOf(shortestLength, i + 1)
        }
    }
    //  println(sums.toList())
    val tree = MaxIndexValueSegmentTree(sums)

    for (i in 1 until n) {
        val threshold = sums[i] - k.toLong()
        val j = tree.findMaxIndex(threshold, 0, i - 1)
     //   println("$j $i $threshold ${sums[i]}")
        if (j >= 0) {
            shortestLength = minOf(shortestLength, i - j)
        }
    }
    return if (shortestLength > n) -1 else shortestLength
}

fun main() {
    println(
        shortestSubarray(
            intArrayOf(-36, 10, -28, -42, 17, 83, 87, 79, 51, -26, 33, 53, 63, 61, 76, 34, 57, 68, 1, -30),
            484
        )
    )
}