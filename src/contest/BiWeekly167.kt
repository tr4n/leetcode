package contest

import local.to2DIntArray
import java.util.*
import kotlin.math.abs

class BiWeekly167 {

    fun scoreBalance(s: String): Boolean {
        val total = s.sumOf { it - 'a' + 1 }

        var left = 0
        for (c in s) {
            left += (c - 'a' + 1)
            if (left * 2 == total) return true
        }
        return false
    }

    fun longestSubarray(nums: IntArray): Int {
        val n = nums.size
        var maxLen = 2
        var current = 2

        for (i in 2 until n) {
            if (nums[i] == nums[i - 1] + nums[i - 2]) {
                current++
                maxLen = maxOf(maxLen, current)
            } else current = 2
        }
        return maxLen
    }


    class SumLongSegmentTree(private val n: Int) {
        private val data = LongArray(n)
        private val tree = LongArray(4 * n)


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

    class DSU(n: Int) {
        private val parent = IntArray(n) { it }
        private val size = IntArray(n) { 1 }

        fun find(u: Int): Int {
            if (u == parent[u]) return u
            parent[u] = find(parent[u])
            return parent[u]
        }

        fun union(a: Int, b: Int) {
            var rootA = find(a)
            var rootB = find(b)
            if (rootA == rootB) return
            if (size[rootA] < size[rootB]) {
                val temp = rootA
                rootA = rootB
                rootB = temp
            }
            parent[rootB] = rootA
            size[rootA] += size[rootB]
        }
    }

    fun maxPartitionFactor(points: Array<IntArray>): Int {
        val n = points.size
        if (n == 2) return 0

        fun distance(a: IntArray, b: IntArray): Long {
            return abs(a[0].toLong() - b[0].toLong()) + abs(a[1].toLong() - b[1].toLong())
        }


        val edges = mutableListOf<Triple<Int, Int, Long>>()

        for (i in 0 until n) {
            val a = points[i]
            for (j in i + 1 until n) {
                val b = points[j]
                val dist = distance(a, b)
                edges.add(Triple(i, j, dist))
            }
        }
        edges.sortWith(compareBy { it.third })

        fun canSplit(d: Long): Boolean {
            val dsu = DSU(2 * n)
            for (edge in edges) {
                val (u, v, dist) = edge
                if (dist >= d) break

                if (dsu.find(u) == dsu.find(v)) return false
                dsu.union(u, v + n)
                dsu.union(v, u + n)

            }
            return true
        }

        var lo = 0L
        var hi = edges.last().third
        var ans = 0L

        while (lo <= hi) {
            val mid = (lo + hi) / 2
            if (canSplit(mid)) {
                ans = mid
                lo = mid + 1
            } else hi = mid - 1
        }

        return ans.toInt()
    }
}

class ExamTracker() {

    private val tree = TreeMap<Int, Long>()

    fun record(time: Int, score: Int) {
        val lower = tree.floorEntry(time)?.value ?: 0L
        tree[time] = lower + score

        for (entry in tree.tailMap(time, false)) {
            tree[entry.key] = entry.value + score
        }
    }

    fun totalScore(startTime: Int, endTime: Int): Long {
        val end = tree.floorEntry(endTime)?.value ?: 0L
        val start = tree.lowerEntry(startTime)?.value ?: 0L
        return end - start
    }

}


fun main() {
    val contest = BiWeekly167()
    println(
        contest.maxPartitionFactor("[[0,0],[0,2],[2,0],[2,2]]".to2DIntArray())
    )
}