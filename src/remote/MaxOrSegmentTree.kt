package remote

class MaxOrSegmentTree(private val data: IntArray) {
    private val n = data.size
    private val tree = IntArray(4 * n)

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = data[l]
            return
        }
        val mid = (l + r) / 2
        build(2 * node, l, mid)
        build(2 * node + 1, mid + 1, r)

        val left = tree[2 * node]
        val right = tree[2 * node + 1]
        tree[node] = left or right
    }

    fun getOR(start: Int, end: Int): Int {
        return query(1, 0, n - 1, start, end)
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int): Int {
        if (l > qr || r < ql) return 0

        if (ql <= l && r <= qr) return tree[node]

        val mid = (l + r) / 2
        val left = query(2 * node, l, mid, ql, qr)
        val right = query(2 * node + 1, mid + 1, r, ql, qr)

        return left or right
    }
}

fun smallestSubarrays(nums: IntArray): IntArray {
    val n = nums.size
    val segmentTree = MaxOrSegmentTree(nums)

    println(segmentTree.getOR(0, n - 1))
    val result = IntArray(n) { n }

    for (start in 0 until n) {
        val target = segmentTree.getOR(start, n - 1)
       // println("$start : $target")
        var low = start
        var high = n - 1
        var index = n - 1

        while (low <= high) {
            val mid = (high + low) / 2
            val currentOr = segmentTree.getOR(start, mid)
         //   println(" - $mid: $currentOr")
            if (currentOr == target) {
                index = mid
                high = mid - 1
            } else {
                low = mid + 1
            }
        }
        result[start] = index - start + 1

    }
    return result
}

fun main() {
    println(
        smallestSubarrays(intArrayOf(1, 0, 2, 1, 3)).toList()
    )
}
