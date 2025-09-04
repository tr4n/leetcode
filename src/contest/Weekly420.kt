package contest

import kotlin.math.sqrt

class Weekly420 {

    fun stringSequence(target: String): List<String> {
        val n = target.length
        var builder = ""
        val result = mutableListOf<String>()

        for (i in 0 until n) {
            val ch = target[i]
            for (c in 'a'..ch) {
                result.add(builder + c)
            }
            builder += ch
        }
        return result
    }

    fun numberOfSubstrings(s: String, k: Int): Int {
        val n = s.length
        val freq = Array(n + 1) { IntArray(26) }

        for (i in 0 until n) {
            for (c in 0 until 26) {
                freq[i + 1][c] = freq[i][c]
            }
            freq[i + 1][s[i] - 'a']++
        }

        fun maxFreq(start: Int, end: Int): Int {
            if (start > end) return 0
            var maxFreq = 0

            for (c in 0 until 26) {
                val f = freq[end + 1][c] - freq[start][c]
                maxFreq = maxOf(maxFreq, f)
            }
            return maxFreq
        }

        var cnt = 0
        for (i in 0 until n) {
            var l = i
            var r = n - 1
            var pos = -1
            while (l <= r) {
                val mid = (l + r) / 2
                val value = maxFreq(i, mid)
                if (value >= k) {
                    pos = mid
                    r = mid - 1
                } else {
                    l = mid + 1
                }
            }
            if (pos < i) continue
            cnt += (n - pos)
        }
        return cnt
    }

    fun minOperations(nums: IntArray): Int {
        val n = nums.size
        if (n == 1) return 0
        val limit = nums.max()
        val isPrime = BooleanArray(limit + 1) { true }
        isPrime[0] = false
        isPrime[1] = false
        for (p in 2..sqrt(limit.toDouble()).toInt()) {
            if (isPrime[p]) {
                for (multiple in p * p..limit step p) {
                    isPrime[multiple] = false
                }
            }
        }

        val divisors = IntArray(limit + 1) { 1 }
        divisors[0] = 0
        for (d in 2..limit / 2) {
            var multiple = 2 * d
            while (multiple <= limit) {
                divisors[multiple] = maxOf(divisors[multiple], d)
                multiple += d
            }
        }

        var greater = nums.last()
        var pos = n - 2
        var cnt = 0
        while (pos >= 0) {
            var num = nums[pos]
            while (num > greater) {
                if (isPrime[num]) return -1
                val d = divisors[num]
                // println("$num / $d")
                num /= d
                cnt++
            }
            greater = num
            pos--
        }
        return cnt
    }

    fun findAnswer(parent: IntArray, s: String): BooleanArray {
        val n = parent.size
        val result = BooleanArray(n)
        val edges = Array(n) { mutableListOf<Int>() }

        for (i in 0 until n) {
            val p = parent[i]
            if (p == -1) continue
            edges[p].add(i)
        }

        val base = 131L
        val mod = 1_000_000_007L
        val pow = LongArray(n + 1)
        pow[0] = 1
        for (i in 1..n) pow[i] = (pow[i - 1] * base) % mod


        class Hash {
            var forward = 0L
            var reverse = 0L
            var len = 0

            fun append(other: Hash) {
                forward = (forward * pow[other.len] + other.forward) % mod
                reverse = (other.reverse * pow[len] + reverse) % mod
                len += other.len
            }

            fun append(c: Char) {
                val code = c.code.toLong()
                forward = (forward * base + code) % mod
                reverse = (reverse + code * pow[len]) % mod
                len++
            }

            fun isPalindrome() = forward == reverse
        }

        fun dfs(node: Int): Hash {
            val hash = Hash()

            for (nextNode in edges[node]) {
                val child = dfs(nextNode)
                hash.append(child)
            }

            hash.append(s[node])
            result[node] = hash.isPalindrome()
            return hash
        }
        dfs(0)
        return result
    }

    fun isAnagram(s: String, t: String): Boolean {
        val map = s.groupingBy { it }.eachCount().toMutableMap()
        for (c in t) {
            val f = map[c]
            if (f == null || f == 0) return false
            map[c] = f - 1
        }
        return map.all { it.value == 0 }
    }
}


fun main() {
    val contest = Weekly420()
    println(
        contest.findAnswer(intArrayOf(-1, 0, 0, 1, 1, 2), "aababa").toList()
    )
}