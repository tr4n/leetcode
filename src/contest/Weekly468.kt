package contest

import remote.SparseTableMinMax
import java.util.PriorityQueue

class Weekly468 {
    fun evenNumberBitwiseORs(nums: IntArray): Int {
        var ans = 0
        for (num in nums) {
            if (num % 2 == 0) ans = ans or num
        }
        return ans
    }

//    fun maxTotalValue(nums: IntArray, k: Int): Long {
//        val min = nums.min().toLong()
//        val max = nums.max().toLong()
//        return k.toLong() * (max - min)
//    }

    fun minSplitMerge(nums1: IntArray, nums2: IntArray): Int {
        val start = nums1.toList()
        val target = nums2.toList()

        if (start == target) return 0

        val seen = mutableSetOf<List<Int>>()
        val queue = ArrayDeque<Pair<List<Int>, Int>>()
        queue.addLast(start to 0)
        seen.add(start)

        while (queue.isNotEmpty()) {
            val (cur, dist) = queue.removeFirst()

            val n = cur.size
            for (l in 0 until n) {
                for (r in l until n) {
                    val sub = cur.subList(l, r + 1)
                    val remain = cur.take(l) + cur.drop(r + 1)

                    for (i in 0..remain.size) {
                        val next = remain.take(i) + sub + remain.drop(i)
                        if (next == target) return dist + 1
                        if (next !in seen) {
                            seen.add(next)
                            queue.addLast(next to dist + 1)
                        }
                    }
                }
            }
        }

        return -1
    }

    fun maxTotalValue(nums: IntArray, k: Int): Long {
        val n = nums.size
        val table = SparseTableMinMax(nums)

        val map = mutableMapOf<Long, Long>()


        for (i in 0 until n) {
             var lo = i
            while (lo < n) {
                var l = lo
                var r = n - 1
                var first = n
                var currDiff = if (lo == i) 0 else table.queryMax(i, lo - 1) - table.queryMin(i, lo - 1)

                while (l <= r) {
                    val mid = (l + r) ushr 1
                    val mx = table.queryMax(i, mid)
                    val mn = table.queryMin(i, mid)
                    val diff = mx - mn
                    if (diff > currDiff) {
                        first = mid
                        r = mid - 1
                    } else {
                        l = mid + 1
                    }
                }

                if (first == n) break
   l = first
                r = n - 1
                var last = first
                val diffAtFirst = table.queryMax(i, first) - table.queryMin(i, first)

                while (l <= r) {
                    val mid = (l + r) ushr 1
                    val mx = table.queryMax(i, mid)
                    val mn = table.queryMin(i, mid)
                    val diff = mx - mn
                    if (diff == diffAtFirst) {
                        last = mid
                        l = mid + 1
                    } else if (diff < diffAtFirst) {
                        l = mid + 1
                    } else {
                        r = mid - 1
                    }
                }

                map[diffAtFirst.toLong()] = (map[diffAtFirst.toLong()] ?: 0L) + (last - first + 1).toLong()
                lo = last + 1
            }
        }

        val keysDesc = map.keys.sortedDescending()
        var taken = 0L
        var ans = 0L
        val need = k.toLong()

        for (value in keysDesc) {
            if (taken >= need) break
            val avail = map[value] ?: 0L
            val use = minOf(need - taken, avail)
            ans += value * use
            taken += use
        }

        return ans
    }

    fun countSuffixSubarrays(i: Int, suffixValue: Int, table: SparseTableMinMax, n: Int): Int {
        var count = 0
        var lo = i
        while (lo < n) {
            var l = lo
            var r = n - 1
            var first = -1

            while (l <= r) {
                val mid = (l + r) ushr 1
                val mx = table.queryMax(i, mid)
                val mn = table.queryMin(i, mid)
                if (mx - mn >= suffixValue) {
                    first = mid
                    r = mid - 1
                } else {
                    l = mid + 1
                }
            }
            if (first == -1) break

            l = first
            r = n - 1
            var last = -1
            while (l <= r) {
                val mid = (l + r) ushr 1
                val mx = table.queryMax(i, mid)
                val mn = table.queryMin(i, mid)
                val diff = mx - mn
                if (diff == suffixValue) {
                    last = mid
                    l = mid + 1
                } else if (diff < suffixValue) {
                    l = mid + 1
                } else {
                    r = mid - 1
                }
            }

            if (last == -1) break
            count += (last - first + 1)
            lo = last + 1
        }
        return count
    }
}

fun main() {
    val contest = Weekly468()
    println(
        contest.maxTotalValue(
            intArrayOf(28, 21, 50, 32),
            4
        )
    )
}