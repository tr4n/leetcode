package remote

class LazySegmentTree(private val n: Int) {
    private val tree = LongArray(4 * n)
    private val lazy = LongArray(4 * n)

    private fun push(node: Int, l: Int, r: Int) {
        if (lazy[node] == 0L) return

        tree[node] += (r - l + 1) * lazy[node]
        if (l != r) {
            lazy[2 * node] += lazy[node]
            lazy[2 * node + 1] += lazy[node]
        }
        lazy[node] = 0
    }

    fun update(node: Int, l: Int, r: Int, ql: Int, qr: Int, value: Long) {
        push(node, l, r)
        if (r < ql || l > qr) return
        if (ql <= l && r <= qr) {
            lazy[node] += value
            push(node, l, r)
            return
        }
        val mid = (l + r) / 2
        update(2 * node, l, mid, ql, qr, value)
        update(2 * node + 1, mid + 1, r, ql, qr, value)
        tree[node] = tree[2 * node] + tree[2 * node + 1]
    }

    fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int): Long {
        push(node, l, r)
        if (r < ql || l > qr) return 0L
        if (ql <= l && r <= qr) return tree[node]
        val mid = (l + r) / 2
        val left = query(2 * node, l, mid, ql, qr)
        val right = query(2 * node + 1, mid + 1, r, ql, qr)
        return left + right
    }

    private fun dfs(node: Int, l: Int, r: Int, output: LongArray) {
        push(node, l, r)
        if (l == r) {
            output[l] = tree[node]
            return
        }
        val mid = (l + r) / 2
        dfs(2 * node, l, mid, output)
        dfs(2 * node + 1, mid + 1, r, output)
    }

    fun toList(): List<Long> {
        val result = LongArray(n)
        dfs(1, 0, n - 1, result)
        return result.toList()
    }

    fun addRange(ql: Int, qr: Int, value: Long) {
        update(1, 0, n - 1, ql, qr, value)
    }
}

fun maxSumRangeQuery(nums: IntArray, requests: Array<IntArray>): Int {
    val n = nums.size
    val mod = 1_000_000_007
    val diff = LongArray(n + 1)

    for (request in requests) {
        val l = request[0]
        val r = request[1]
        diff[l]++
        if (r + 1 <= n) {
            diff[r + 1]--
        }
    }

    val priorities = LongArray(n)
    var currentCount = 0L
    for (i in 0 until n) {
        currentCount += diff[i]
        priorities[i] = currentCount
    }

    val orders = (0 until n).sortedBy { priorities[it] }
    nums.sort()
    val list = LongArray(n)
    var id = 0
    for (i in orders) list[i] = nums[id++].toLong()

    val prefix = LongArray(n + 1)
    for (i in 0 until n) prefix[i + 1] = prefix[i] + list[i]

    var ans = 0L
    for (request in requests) {
        val (l, r) = request
        val sum = prefix[r + 1] - prefix[l]
        ans = (ans + sum) % mod
    }
    return ans.toInt()
}