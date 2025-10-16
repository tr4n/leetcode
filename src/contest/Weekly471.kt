package contest

class Weekly471 {

    fun longestBalanced(s: String): Int {
        val n = s.length
        var ans = 0

        for (limit in 1..2) {
            var distinct = 0
            var l = 0
            val freq = IntArray(3)

            for (r in 0 until n) {
                val id = s[r] - 'a'
                if (freq[id] == 0) distinct++
                freq[id]++

                while (distinct > limit) {
                    val left = s[l] - 'a'
                    freq[left]--
                    if (freq[left] == 0) distinct--
                    l++
                }
                // println("${s.substring(l, r +1)} ${freq.toList()}")
                val frequents = freq.filter { it > 0 }
                val uniqueFreqs = frequents.toSet().toList()
                var isBalanced = false

                if (uniqueFreqs.size == 1) {
                    isBalanced = true
                } else if (uniqueFreqs.size == 2) {
                    val freqsList = uniqueFreqs
                    val f1 = freqsList[0]
                    val f2 = freqsList[1]

                    val countF1 = frequents.count { it == f1 }
                    val countF2 = frequents.count { it == f2 }

                    if ((f1 == 1 && countF1 == 1) || (f2 == 1 && countF2 == 1)) {
                        isBalanced = true
                    } else if ((f1 == f2 + 1 && countF1 == 1) || (f2 == f1 + 1 && countF2 == 1)) {
                        isBalanced = true
                    }
                }

                if (isBalanced) {
                    ans = maxOf(ans, r - l + 1)
                }

            }
        }

        return ans
    }

    fun sumOfAncestors(n: Int, edges: Array<IntArray>, nums: IntArray): Long {

        fun getSquareFree(num: Int): Int {
            var n = num
            var d = 2
            while (d * d <= n) {
                while (n % (d * d) == 0) {
                    n /= (d * d)
                }
                d++
            }
            return n
        }

        val graph = Array(n) { mutableListOf<Int>() }
        for ((u, v) in edges) {
            graph[u].add(v)
            graph[v].add(u)
        }

        val squareFrees = IntArray(n)
        for (i in 0 until n) {
            squareFrees[i] = getSquareFree(nums[i])
        }

        var totalSum = 0L
        val counts = mutableMapOf<Int, Long>()

        fun dfs(u: Int, p: Int) {
            val uVal = squareFrees[u]
            val uCount = counts[uVal] ?: 0
            totalSum += uCount
            counts[uVal] = uCount + 1L

            for (v in graph[u]) {
                if (v == p) continue
                dfs(v, u)
            }
            counts[uVal] = (counts[uVal] ?: 0L) - 1
        }

        dfs(0, -1)
        return totalSum
    }

    fun magicalSum(m: Int, k: Int, nums: IntArray): Int {
        val mod = 1_000_000_007L
        val n = nums.size

        val limit = 1 shl m
        var dp = Array(limit) { LongArray(k + 1) }
        dp[0][0] = 1L

        for(i in 0 until m) {
            val next = Array(limit) { LongArray(k + 1) }
            for(mask in 0 until limit) {
                for(bits in 0..k) {
                    if(dp[mask][bits] == 0L) continue
                    for(j in 0 until n) {
                        val newMask = mask + (1 shl j)
                        val newBits = bits + newMask.countOneBits()
                        if(newBits <= k) {
                            next[newMask][newBits] = (next[newMask][newBits] + dp[mask][bits]) % mod
                        }
                    }
                }
            }
            dp = next
        }
        return dp[0][k].toInt()
    }
}

fun main() {
    val contest = Weekly471()
    println(
        contest.longestBalanced("aba")

    )
}