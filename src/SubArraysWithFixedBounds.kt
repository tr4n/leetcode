class SubArraysWithFixedBounds(private val arr: IntArray) {
    private val n = arr.size
    private val log = IntArray(n + 1)
    private val st: Array<Array<Pair<Int, Int>>>

    init {
        // log2
        for (i in 2..n) {
            log[i] = log[i / 2] + 1
        }

        val K = log[n] + 1
        st = Array(n) { Array(K) { Int.MAX_VALUE to Int.MIN_VALUE } }

        // base
        for (i in 0 until n) st[i][0] = arr[i] to arr[i]

        // build
        for (j in 1 until K) {
            var i = 0
            while (i + (1 shl j) <= n) {
                val left = st[i][j - 1]
                val right = st[i + (1 shl (j - 1))][j - 1]
                val max = maxOf(left.second, right.second)
                val min = minOf(left.first, right.first)
                st[i][j] = min to max
                i++
            }
        }
    }

    // Query [l,r] in O(1)
    fun query(l: Int, r: Int): Pair<Int, Int> {
        val j = log[r - l + 1]
        val left = st[l][j]
        val right = st[r - (1 shl j) + 1][j]
        val max = maxOf(left.second, right.second)
        val min = minOf(left.first, right.first)
        return min to max
    }
}

fun countSubarrays(nums: IntArray, minK: Int, maxK: Int): Long {
    val n = nums.size
    val table = SubArraysWithFixedBounds(nums)

    fun findBoundary(
        left: Int,
        right: Int,
        target: Int,
        findLeft: Boolean,
        increasing: Boolean,
        queryFn: (Int) -> Int,
    ): Int {
        var l = left
        var r = right
        var answer = -1

        while (l <= r) {
            val mid = (l + r) / 2
            val value = queryFn(mid)
            if(increasing && !findLeft) {
              //  println("$l $r $mid $value ")
            }
            when {
                value == target -> {
                    answer = mid
                    if (findLeft) r = mid - 1 else l = mid + 1
                }

                increasing && value < target -> l = mid + 1
                increasing && value > target -> r = mid - 1
                !increasing && value > target -> l = mid + 1
                !increasing && value < target -> r = mid - 1
            }
        }

        return answer
    }
  //  println(nums.indices.toList())
  //  println(nums.toList())
    var cnt = 0L
    for (i in 0 until n) {
        if (nums[i] !in minK..maxK) continue
        val maxStart = findBoundary(i, n - 1, maxK, findLeft = true, increasing = true) { mid ->
            table.query(i, mid).second
        }
        if (maxStart == -1) continue

        val maxEnd = findBoundary(i, n - 1, maxK, findLeft = false, increasing = true) { mid ->
            table.query(i, mid).second
        }
        if (maxEnd == -1) continue

        val minStart = findBoundary(i, n - 1, minK, findLeft = true, increasing = false) { mid ->
            table.query(i, mid).first
        }
        if (minStart == -1) continue

        val minEnd = findBoundary(i, n - 1, minK, findLeft = false, increasing = false) { mid ->
            table.query(i, mid).first
        }
        if (minEnd == -1) continue

      //  println("${i} ${maxStart}-${maxEnd}, $minStart-$minEnd")

        val start = maxOf(minStart, maxStart)
        val end = minOf(maxEnd, minEnd)
        val count = if (end < start) 0 else (end - start + 1)
        cnt += count.toLong()
    }
    return cnt
}

fun main() {
    println(
        countSubarrays(
            intArrayOf(1, 3, 5, 2, 7, 5),
            1, 5
        )
    )
}