package contest

import java.util.*

class BiWeekly166 {
    fun majorityFrequencyGroup(s: String): String {
        val freq = s.groupingBy { it }.eachCount()

        val groups = mutableMapOf<Int, MutableList<Char>>()
        for ((c, f) in freq) {
            groups.computeIfAbsent(f) { mutableListOf() }.add(c)
        }

        var bestFreq = -1
        var bestGroup: List<Char> = emptyList()
        for ((f, chars) in groups) {
            if (chars.size > bestGroup.size ||
                (chars.size == bestGroup.size && f > bestFreq)
            ) {
                bestFreq = f
                bestGroup = chars
            }
        }

        return bestGroup.joinToString("")
    }

    fun climbStairs(n: Int, costs: IntArray): Int {
        val dp = IntArray(n + 1) { Int.MAX_VALUE }
        dp[0] = 0

        for (i in 1..n) {
            dp[i] = dp[i - 1] + costs[i - 1] + 1
            if (i > 1) dp[i] = minOf(dp[i], dp[i - 2] + costs[i - 1] + 4)
            if (i > 2) dp[i] = minOf(dp[i], dp[i - 3] + costs[i - 1] + 9)
        }
        return dp[n]
    }

    fun distinctPoints(s: String, k: Int): Int {
        val n = s.length
        val prefixX = IntArray(n + 1)
        val prefixY = IntArray(n + 1)

        for (i in 0 until n) {
            prefixX[i + 1] = prefixX[i]
            prefixY[i + 1] = prefixY[i]
            when (s[i]) {
                'U' -> prefixY[i + 1]++
                'D' -> prefixY[i + 1]--
                'L' -> prefixX[i + 1]--
                'R' -> prefixX[i + 1]++
            }
        }
        val totalX = prefixX[n]
        val totalY = prefixY[n]
        val set = mutableSetOf<Pair<Int, Int>>()

        for (i in 0 until n - k + 1) {
            val x = prefixX[i + k] - prefixX[i]
            val y = prefixY[i + k] - prefixY[i]
            set.add(
                Pair(
                    totalX - x,
                    totalY - y
                )
            )
        }
        return set.size
    }

    fun maxAlternatingSum(nums: IntArray, swaps: Array<IntArray>): Long {
        val n = nums.size
        val parent = IntArray(n) { it }
        val size = IntArray(n) { 1 }

        fun find(u: Int): Int {
            if (u == parent[u]) return u
            val root = find(parent[u])
            parent[u] = root
            return root
        }

        fun union(a: Int, b: Int): Boolean {
            val rootA = find(a)
            val rootB = find(b)

            if (rootA == rootB) return false
            if (size[rootA] > size[rootB]) {
                size[rootA] += size[rootB]
                parent[rootB] = rootA
            } else {
                size[rootB] += size[rootA]
                parent[rootA] = rootB
            }
            return true
        }

        for ((a, b) in swaps) {
            union(a, b)
        }

        val indexGroups = mutableMapOf<Int, IntArray>()
        val valueGroups = mutableMapOf<Int, MutableList<Int>>()
        for (i in 0 until n) {
            val root = find(i)
            indexGroups.computeIfAbsent(root) { intArrayOf(0, 0) }[i % 2]++
            valueGroups.computeIfAbsent(root) { mutableListOf() }.add(nums[i])
        }

        valueGroups.forEach { it.value.sort() }

        var ans = 0L
        for ((root, counts) in indexGroups) {
            val values = valueGroups[root] ?: continue
            val (even, odd) = counts
            for (i in 0 until even) ans += values.removeLastOrNull() ?: 0
            for (i in 0 until odd) ans -= values.removeLastOrNull() ?: 0
        }
        return ans

    }
}

fun main() {
    val contest = BiWeekly166()
    println(
        contest
    )
}