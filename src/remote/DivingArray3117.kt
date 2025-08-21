package remote

fun findAndBounds(table: ANDSparseTable, left: Int, right: Int, k: Int): Pair<Int, Int>? {
    var l = left
    var r = right
    var startIndex = -1
    while (l <= r) {
        val mid = (l + r) / 2
        val value = table.queryAND(mid, right)
        if (value >= k) {
            if (value == k) startIndex = mid
            r = mid - 1
        } else {
            l = mid + 1
        }
    }

    l = left
    r = right
    var endIndex = -1
    while (l <= r) {
        val mid = (l + r) / 2
        val value = table.queryAND(mid, right)
        if (value <= k) {
            if (value == k) endIndex = mid
            l = mid + 1
        } else {
            r = mid - 1
        }
    }

    return if (startIndex != -1 && endIndex != -1 && startIndex <= endIndex) {
        startIndex to endIndex
    } else null
}

fun minimumValueSum(nums: IntArray, andValues: IntArray): Int {
    val n = nums.size
    val m = andValues.size
    val andTable = ANDSparseTable(nums)


    val dp = Array(m + 1) { IntArray(n + 1) { Int.MAX_VALUE } }
    dp[0][0] = 0

    for (j in 1..m) {
        val minTree = MinWithIndexSegmentTree(dp[j - 1])

        for (i in 1..n) {
            val bounds = findAndBounds(andTable, 0, i - 1, andValues[j - 1]) ?: continue
            val (startIndex, endIndex) = bounds
            if (startIndex > endIndex) continue

            val (minValue, minIndex) = minTree.query(startIndex, endIndex)
            if (minValue != Int.MAX_VALUE) {
                dp[j][i] = minValue + nums[i - 1]
            }
        }
        //  println(dp[j].toList())
    }
    //  println(dp.joinToString("\n") { it.toList().toString() })
    return if (dp[m][n] == Int.MAX_VALUE) -1 else dp[m][n]
}

fun main() {
    // println(4 and 8)
    //   println(4 and 8 and 9)
    println(
        minimumValueSum(
            intArrayOf(1, 4, 3, 3, 2),
            intArrayOf(0, 3, 3, 2)
        )
    )
}