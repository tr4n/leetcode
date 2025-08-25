package remote

import local.to2DIntArray
import java.util.*

fun findDuplicates(nums: IntArray): List<Int> {
    val n = nums.size
    val result = mutableListOf<Int>()
    for (i in 0 until n) {
        var currentNum = nums[i]
        while (currentNum != i + 1) {
            val correctPos = currentNum - 1
            if (nums[correctPos] == currentNum) break

            val temp = nums[correctPos]
            nums[correctPos] = currentNum
            nums[i] = temp
            currentNum = temp
        }

    }
    for (i in 0 until n) {
        if (nums[i] != i + 1) {
            result.add(nums[i])
        }
    }
    return result
}

fun waysToReachTarget(target: Int, types: Array<IntArray>): Int {
    val mod = 1_000_000_007
    val n = types.size

    val dp = Array(n) { LongArray(target + 1) }
    val (count0, mark0) = types[0]
    for (j in 0..count0) {
        val score = j * mark0
        if (score > target) break
        dp[0][score] = 1
    }
    for (i in 1 until n) {
        val (count, mark) = types[i]
        for (j in 0..count) {
            val score = j * mark
            for (p in score..target) {
                val d = dp[i - 1][p - score]
                dp[i][p] += d
                dp[i][p] = dp[i][p] % mod
            }
        }
    }
    // println(dp.print())
    return (dp[n - 1][target] % mod).toInt()
}

fun sumCounts(nums: IntArray): Int {
    val mod = 1_000_000_007
    val n = nums.size
    val lastSeen = mutableMapOf<Int, Int>()

    val dp = LongArray(n)
    dp[0] = 1L
    lastSeen[nums[0]] = 0
    var d = 1L
    var squares = 1L
    for (i in 1 until n) {
        val num = nums[i]
        val lastIndex = lastSeen[num] ?: -1
        lastSeen[num] = i
        d += (i - lastIndex).toLong()
        dp[i] = d
        squares += d
        squares %= mod
    }
    squares = (squares * squares) % mod
    println(dp.toList())
    return (squares % mod).toInt()
}

fun fullBloomFlowers(flowers: Array<IntArray>, people: IntArray): IntArray {
    flowers.sortBy { it[0] }
    val map = TreeMap<Int, Int>()
    for ((start, end) in flowers) {
        map[start] = (map[start] ?: 0) + 1
        map[end + 1] = (map[end + 1] ?: 0) - 1
    }
    var count = 0
    map.forEach { (time, delta) ->
        count += delta
        map[time] = count
    }
    // println(map.toList().joinToString("\n"))
    return IntArray(people.size) {
        val time = people[it]
        val count = map.floorEntry(time)?.value ?: 0
        maxOf(count, 0)
    }
}

fun minInterval(intervals: Array<IntArray>, queries: IntArray): IntArray {
    val queryList = queries.withIndex().sortedBy { it.value }
  //  intervals.sortWith(comparator = compareBy<IntArray> { it[1] }.thenByDescending { it[0] })
    val minQuery = queryList.first().value
    val maxQuery = queryList.last().value
    val intervalList = intervals.filter {
        it[0] <= maxQuery && it[1] >= minQuery
    }.sortedWith(compareBy({ it[1] }, { -it[0] }))

    val answers = IntArray(queries.size) { -1 }
    var i = 0
    for (query in queryList) {

        while (i < intervalList.size) {
            val (start, end) = intervalList[i]
            if (query.value in start..end) {
                answers[query.index] = end - start + 1
                break
            } else i++
        }
    }
    return answers
}

fun main() {
    println(
        minInterval(
            "[[1,4],[2,4],[3,6],[4,4]]".to2DIntArray(),
            intArrayOf(2, 3, 4, 5)
        ).toList()
    )
}