package remote

class MergeSetTree<T : Comparable<T>>(private val arr: List<T>) {
    private val n = arr.size
    private val tree = Array(4 * n) { setOf<T>() }

    init {
        if (arr.isNotEmpty()) build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = setOf(arr[l])
            return
        }
        val mid = (l + r) / 2
        build(node * 2, l, mid)
        build(node * 2 + 1, mid + 1, r)
        tree[node] = tree[node * 2] + tree[node * 2 + 1]
    }

    fun query(ql: Int, qr: Int): Set<T> {
        return query(1, 0, n - 1, ql, qr)
    }

    fun count(ql: Int, qr: Int, onCount: (Set<T>) -> Long): Long {
        return queryCount(1, 0, n - 1, ql, qr, onCount)
    }

    private fun queryCount(node: Int, l: Int, r: Int, ql: Int, qr: Int, onCount: (Set<T>) -> Long): Long {
        if (qr < l || r < ql) return 0
        if (ql <= l && r <= qr) return onCount(tree[node])
        val mid = (l + r) / 2
        val left = queryCount(node * 2, l, mid, ql, qr, onCount)
        val right = queryCount(node * 2 + 1, mid + 1, r, ql, qr, onCount)
        return left + right
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int): Set<T> {
        if (qr < l || r < ql) return emptySet()
        if (ql <= l && r <= qr) return tree[node]
        val mid = (l + r) / 2
        val left = query(node * 2, l, mid, ql, qr)
        val right = query(node * 2 + 1, mid + 1, r, ql, qr)
        return left + right
    }
}

class DiffCountSegmentTree(private val arr: IntArray) {
    private val n = arr.size
    private val tree = IntArray(4 * n)

    init {
        if (arr.isNotEmpty()) build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = 1
            return
        }
        val mid = (l + r) / 2
        build(node * 2, l, mid)
        build(node * 2 + 1, mid + 1, r)
        tree[node] = tree[node * 2] + tree[node * 2 + 1]
    }

    fun countDiff(ql: Int, qr: Int): Int {
        return queryCount(1, 0, n - 1, ql, qr)
    }

    private fun queryCount(node: Int, l: Int, r: Int, ql: Int, qr: Int): Int {
        if (qr < l || r < ql) return 0
        if (ql <= l && r <= qr) return tree[node]
        val mid = (l + r) / 2
        val left = queryCount(node * 2, l, mid, ql, qr)
        val right = queryCount(node * 2 + 1, mid + 1, r, ql, qr)
        return left + right
    }
}

fun subarraysWithKDistinct(nums: IntArray, k: Int): Int {
    val n = nums.size
    val set = mutableSetOf<Int>()
    var left = 0
    var cnt = 0
    val max = nums.max()
    val freq = IntArray(max + 1)

    for (right in 0 until n) {

        val num = nums[right]
        if (set.size == k && num !in set) {
            while (set.size == k) {
                println(nums.toList().subList(left, right))
                cnt += (right - left - k + 1)
                val leftMost = nums[left]
                freq[leftMost]--
                if (freq[leftMost] <= 0) set.remove(nums[left])
                left++
            }
        }

        set.add(num)
        freq[num]++
    }
    while (set.size == k) {
        println(nums.toList().subList(left, n))
        cnt += (n - left - k + 1)
        val leftMost = nums[left]
        freq[leftMost]--
        if (freq[leftMost] <= 0) set.remove(nums[left])
        left++
    }

    return cnt
}

fun main() {
    println(
        subarraysWithKDistinct(
            intArrayOf(7, 9, 6, 10, 3, 7, 6, 14, 9, 14, 7, 6, 13, 5),
            4
        )
    )
}