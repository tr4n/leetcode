package contest

import remote.MonotonicStackInt
import java.util.Stack

class Weekly466 {
    fun minOperations1(nums: IntArray): Int {
        val isSame = nums.toSet().size == 1
        return if (isSame) 0 else 1
    }

    class DSU(n: Int) {
        private val parent = IntArray(n) { it }

        fun find(i: Int): Int {
            if (parent[i] != i) {
                parent[i] = find(parent[i])
            }
            return parent[i]
        }

        fun union(i: Int, j: Int): Boolean {
            val rootI = find(i)
            val rootJ = find(j)
            if (rootI != rootJ) {
                parent[rootI] = rootJ
                return true
            }
            return false
        }
    }

    fun charToInt(c: Char): Int = c - 'a'

    fun minOperationsDSU(s: String): Int {
        val dsu = DSU(26)
        var operations = 0

        val uniqueChars = s.toSet()

        for (ch in uniqueChars) {
            if (ch == 'a') continue

            var c = ch
            while (c != 'a') {
                val nextCharVal = (charToInt(c) + 1) % 26
                val nextChar = ('a' + nextCharVal)

                val u = charToInt(c)
                val v = charToInt(nextChar)

                if (dsu.union(u, v)) {
                    operations++
                }

                c = nextChar
            }
        }

        return operations
    }

    fun minOperations(s: String): Int {
        val min = s.filter { it != 'a' }.minOrNull() ?: 'a'
        return if (min == 'a') 0 else 'z' + 1 - min
    }

    class SegmentTree(private val arr: IntArray) {
        private val n = arr.size
        private val tree = IntArray(4 * n)

        init {
            build(1, 0, n - 1)
        }

        private fun build(node: Int, l: Int, r: Int) {
            if (l == r) {
                tree[node] = arr[l]
                return
            }
            val mid = (l + r) / 2
            build(node * 2, l, mid)
            build(node * 2 + 1, mid + 1, r)
            tree[node] = maxOf(tree[node * 2], tree[node * 2 + 1])
        }

        fun queryMax(qL: Int, qR: Int, node: Int = 1, l: Int = 0, r: Int = n - 1): Int {
            if (qR < l || r < qL) return Int.MIN_VALUE
            if (qL <= l && r <= qR) return tree[node]
            val mid = (l + r) / 2
            return maxOf(
                queryMax(qL, qR, node * 2, l, mid),
                queryMax(qL, qR, node * 2 + 1, mid + 1, r)
            )
        }
    }

    fun bowlSubarrays1(nums: IntArray): Long {
        val n = nums.size
        val mono = MonotonicStackInt(nums)
        val geLeft = mono.greaterEqualLeft()
        val geRight = mono.greaterEqualRight()
        var result = 0L

        for (i in 1 until n - 1) {
            val l = geLeft[i]
            val r = geRight[i]

            if (l !in 0 until n || r !in 0 until n) continue
            val len = r - l + 1
            if (len < 3) continue
            result++
        }

        return result
    }

    fun bowlSubarrays(nums: IntArray): Long {
        val n = nums.size
        val gRight = IntArray(n) { n }

        val stack = Stack<Int>()
        for (i in 0 until n) {
            while (stack.isNotEmpty() && nums[stack.peek()] < nums[i]) {
                val top = stack.pop()
                gRight[top] = i
            }
            stack.add(i)
        }
        stack.clear()

        val gLeft = IntArray(n) { -1 }
        for (i in (n - 1) downTo 0) {
            while (stack.isNotEmpty() && nums[stack.peek()] < nums[i]) {
                val top = stack.pop()
                gLeft[top] = i
            }
            stack.add(i)
        }


        var result = 0L
        for (i in 1 until n - 1) {
            val r = gRight[i]
            val l = gLeft[i]
            val len = r - l + 1
            if (r in 0 until n && l in 0 until n && len >= 3) {
                result++
            }
        }

        return result
    }

    fun countBinaryPalindromes(n: Long): Int {
        if (n == 0L) return 1
        var ans = 1

        val maxLen = 64 - n.countLeadingZeroBits()


        for (len in 1 until maxLen) {
            val halfLen = (len + 1) / 2
            ans += 1 shl (halfLen - 1)
        }

        val halfLen = (maxLen + 1) / 2
        val start = 1L shl (halfLen - 1)
        val prefix = n shr (maxLen - halfLen)

        if (prefix >= start) {
            ans += (prefix - start).toInt()
            val pal = buildPalindrome(prefix, maxLen)
            if (pal <= n) ans++
        }

        return ans
    }

    private fun buildPalindrome(half: Long, len: Int): Long {
        var pal = half
        var x = if (len % 2 == 1) half shr 1 else half
        while (x > 0) {
            pal = (pal shl 1) or (x and 1L)
            x = x shr 1
        }
        return pal
    }
}

fun main() {
    val contest = Weekly466()
    print(
        contest.countBinaryPalindromes(2147483648)
    )
}