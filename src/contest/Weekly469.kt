package contest

import kotlin.math.abs

class Weekly469 {
    fun largestPerimeter(nums: IntArray): Int {
        val n = nums.size
        // a < b < c
        // c - a < b < a + c
        return 0
    }

    fun decimalRepresentation(n: Int): IntArray {
        var base = 1
        var num = n
        val result = mutableListOf<Int>()
        while (num > 0) {
            val x = (num % 10) * base
            if (x > 0) result.add(x)
            num /= 10
            base *= 10
        }
        result.sortDescending()
        return result.toIntArray()
    }

    fun splitArray(nums: IntArray): Long {
        val n = nums.size
        val prefix = LongArray(n + 1)
        for (i in 0 until n) prefix[i + 1] = prefix[i] + nums[i].toLong()
        val total = prefix[n]

        val isInc = BooleanArray(n)
        isInc[0] = true
        for (i in 1 until n) isInc[i] = isInc[i - 1] && (nums[i] > nums[i - 1])

        val isDec = BooleanArray(n)
        isDec[n - 1] = true
        for (i in (n - 2) downTo 0) isDec[i] = isDec[i + 1] && (nums[i] > nums[i + 1])

        var ans = Long.MAX_VALUE
        for (i in 1 until n) {
            if (isInc[i - 1] && isDec[i]) {
                val left = prefix[i]
                val right = total - prefix[i]
                ans = minOf(ans, abs(left - right))
            }
        }
        return if (ans == Long.MAX_VALUE) -1 else ans
    }


    class SumLongSegmentTree(nums: List<Long>) {
        private val n = nums.size
        private val data = nums
        private val tree = LongArray(4 * n)

        init {
            build(1, 0, n - 1)
        }

        private fun build(node: Int, l: Int, r: Int) {
            if (l == r) {
                tree[node] = data[l]
            } else {
                val mid = (l + r) / 2
                build(node * 2, l, mid)
                build(node * 2 + 1, mid + 1, r)
                tree[node] = tree[node * 2] + tree[node * 2 + 1]
            }
        }


        private fun query(node: Int, l: Int, r: Int, i: Int, j: Int): Long {
            if (r < i || l > j) return 0

            if (i <= l && r <= j) return tree[node]

            val mid = (l + r) / 2
            val left = query(node * 2, l, mid, i, j)
            val right = query(node * 2 + 1, mid + 1, r, i, j)
            return left + right
        }

        private fun update(node: Int, l: Int, r: Int, idx: Int, value: Long) {
            if (l == r) {
                tree[node] = value
            } else {
                val mid = (l + r) / 2
                if (idx <= mid) {
                    update(node * 2, l, mid, idx, value)
                } else {
                    update(node * 2 + 1, mid + 1, r, idx, value)
                }
                tree[node] = tree[node * 2] + tree[node * 2 + 1]
            }
        }

        fun update(index: Int, value: Long) {
            update(1, 0, n - 1, index, value)
        }

        fun sumRange(left: Int, right: Int): Long {
            return query(1, 0, n - 1, left, right)
        }

    }

    fun zigZagArrays1(n: Int, l: Int, r: Int): Int {
        val mod = 1_000_000_007
        val m = r - l + 1

        var up = LongArray(m)
        var down = LongArray(m)

        for (k in 0 until m) {
            val value = l + k
            up[k] = k.toLong()
            down[k] = (r - value).toLong()
        }

        for (i in 3..n) {
            val newDpUp = LongArray(m)
            val newDpDown = LongArray(m)

            var sumDown = 0L
            for (k in 0 until m) {
                newDpUp[k] = sumDown
                sumDown = (sumDown + down[k]) % mod
            }

            var sumUp = 0L
            for (k in m - 1 downTo 0) {
                newDpDown[k] = sumUp
                sumUp = (sumUp + up[k]) % mod
            }

            up = newDpUp
            down = newDpDown
        }

        var ans = 0L
        for (i in 0 until m) {
            ans = (ans + up[i]) % mod
            ans = (ans + down[i]) % mod
        }
        return ans.toInt()
    }

    fun zigZagArrays(n: Int, l: Int, r: Int): Int {
        val mod = 1_000_000_007L
        val m = (r - l + 1).toLong()
        if (n == 1) return (m % mod).toInt()

        fun modPow(base: Long, exp: Long, mod: Long): Long {
            var b = base % mod
            var e = exp
            var res = 1L
            while (e > 0) {
                if ((e and 1L) == 1L) res = (res * b) % mod
                b = (b * b) % mod
                e = e shr 1
            }
            return res
        }

        val ans = (m % mod) * modPow(m - 1, (n - 1).toLong(), mod) % mod
        return ans.toInt()
    }

    fun matrixMultiply(a: Array<LongArray>, b: Array<LongArray>): Array<LongArray> {
        val mod = 1_000_000_007
        val size = a.size
        val result = Array(size) { LongArray(size) }
        for (i in 0 until size) {
            for (j in 0 until size) {
                for (k in 0 until size) {
                    result[i][j] = (result[i][j] + a[i][k] * b[k][j]) % mod
                }
            }
        }
        return result
    }

    fun matrixPower(base: Array<LongArray>, exp: Long): Array<LongArray> {
        var p = exp
        val size = base.size
        var result = Array(size) { LongArray(size) }
        for (i in 0 until size) {
            result[i][i] = 1L
        }

        var a = base
        while (p > 0) {
            if (p % 2 == 1L) {
                result = matrixMultiply(result, a)
            }
            a = matrixMultiply(a, a)
            p /= 2
        }
        return result
    }


}

fun main() {
    val contest = Weekly469()
    println(contest.zigZagArrays(3,1,3))
}