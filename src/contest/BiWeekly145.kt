package contest

import java.util.*
import kotlin.math.sqrt

class BiWeekly145 {
    fun minOperations(n: Int, m: Int): Int {
        val limit = minOf(10 * n, 10 * m, 100_000)
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

        fun neighborsNonPrime(n: Int): List<Int> {
            val res = mutableListOf<Int>()
            var pow10 = 1
            var x = n
            while (x > 0) {
                val digit = x % 10
                if (digit > 0) {
                    val candidate = n - pow10
                    if (!isPrime[candidate]) res.add(candidate)
                }
                if (digit < 9) {
                    val candidate = n + pow10
                    if (!isPrime[candidate]) res.add(candidate)
                }
                x /= 10
                pow10 *= 10
            }
            return res
        }

        if (isPrime[n] || isPrime[m]) return -1

        val d = IntArray(limit + 1) { Int.MAX_VALUE }
        d[n] = n
        val pq = PriorityQueue<Pair<Int, Int>>(compareBy { it.second })
        pq.add(n to n)

        while (pq.isNotEmpty()) {
            val (num, currentCost) = pq.poll()

            if (num == m) {
                return currentCost
            }

            val neighbors = neighborsNonPrime(num)
            for (neighbor in neighbors) {
                val newCost = currentCost + neighbor
                if (newCost < d[neighbor]) {
                    d[neighbor] = newCost
                    pq.add(neighbor to newCost)
                }
            }
        }
        return -1
    }

    fun findMinimumTime(strength: List<Int>, k: Int): Int {
        val tree = TreeMap<Int, Int>()
        for (num in strength) {
            tree[num] = (tree[num] ?: 0) + 1
        }
        val n = strength.size
        fun dfs(mask: Int): Int {
            val unlockedCount = Integer.bitCount(mask)
            if (unlockedCount == n) return 0

            val x = 1 + unlockedCount * k
            var best = Int.MAX_VALUE

            for (i in 0 until n) {
                if (mask and (1 shl i) != 0) continue
                val time = (strength[i] + x - 1) / x
                val nextMask = mask or (1 shl i)
                val nextTime = dfs(nextMask)
                best = minOf(best, time + nextTime)
            }
            return best
        }
        return dfs(0)
    }
}

fun main() {
    val contest = BiWeekly145()
    println(
        contest.findMinimumTime(listOf(7, 3, 6, 18, 22, 50), 4)
    )
}