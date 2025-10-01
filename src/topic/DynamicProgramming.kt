package topic

import kotlin.math.abs

fun maxSumAfterPartitioning(arr: IntArray, k: Int): Int {
    val n = arr.size
    val dp = IntArray(n + 1)

    for (i in 1..n) {
        var lastMax = -1
        for (j in 1..k) {
            if (i < j) break
            lastMax = maxOf(lastMax, arr[i - j])
            dp[i] = maxOf(dp[i], dp[i - j] + lastMax * j)
        }
    }
    return dp[n]
}

fun longestStrChain(words: Array<String>): Int {
    words.sortBy { it.length }
    val n = words.size
    val dp = IntArray(n) { 1 }

    fun isPredecessor(a: String, b: String): Boolean {
        if (a.length != b.length - 1) return false
        var i = 0
        var j = 0
        while (i < a.length && j < b.length && a[i] == b[j]) {
            i++
            j++
        }
        j++
        while (i < a.length && j < b.length && a[i] == b[j]) {
            i++
            j++
        }
        return i == a.length && j == b.length
    }

    for (i in 1 until n) {
        for (j in 0 until i) {
            if (!isPredecessor(words[j], words[i])) continue
            dp[i] = maxOf(dp[i], dp[j] + 1)
        }
    }
    //  println(isPredecessor("a", "ab"))
    return dp.max()
}

fun numWaterBottles(numBottles: Int, numExchange: Int): Int {

    fun change(filled: Int, empty: Int): Int {
        //  println("$filled, $empty")
        if (filled == 0) return 0
        val exchanged = (filled + empty) / numExchange
        val emptyBottles = (filled + empty) % numExchange
        return filled + change(exchanged, emptyBottles)
    }
    return change(numBottles, 0)
}

fun maxBottlesDrunk(numBottles: Int, numExchange: Int): Int {
    fun change(emptyBottles: Int, exchangeNum: Int): Int {
        //  println("$filled, $empty")
        if (emptyBottles < exchangeNum) return 0
        return 1 + change(1 + emptyBottles - exchangeNum, exchangeNum + 1)
    }

    var ans = numBottles
    var numEmpties = numBottles
    var exchangeNum = numExchange

    while (numEmpties >= exchangeNum) {
        ans++
        numEmpties += 1 - exchangeNum
        exchangeNum++
    }

    return ans
}

fun numRollsToTarget(n: Int, k: Int, target: Int): Int {
    val mod = 1_000_000_007
    val dp = Array(n + 1) { LongArray(target + 1) }

    // for (i in 0..n) dp[i][0] = 1L
    dp[0][0] = 1L
    for (t in 1..target) {
        for (j in 1..minOf(k, t)) {
            for (i in 1..n) {
                dp[i][t] += dp[i - 1][t - j]
                dp[i][t] %= mod
            }
        }
    }
    return dp[n][target].toInt()
}

fun maxDistance(grid: Array<IntArray>): Int {
    val m = grid.size
    val n = grid[0].size
    var ans = 0
    val dx = intArrayOf(0, 0, 1, -1)
    val dy = intArrayOf(1, -1, 0, 0)
    val queue = ArrayDeque<Pair<Int, Int>>()
    for (i in 0 until m) {
        for (j in 0 until n) {
            if (grid[i][j] == 1) queue.addLast(i to j)
        }
    }

    if (queue.isEmpty() || queue.size == m * n) return -1

    while (true) {
        var numCells = queue.size
        while (numCells-- > 0) {
            val (r, c) = queue.removeFirst()

            for (i in 0 until 4) {
                val x = r + dx[i]
                val y = c + dy[i]
                if (x !in 0 until m || y !in 0 until m) continue
                if (grid[x][y] == 1) continue
                grid[x][y] = 1
                queue.addLast(x to y)
            }
        }
        if (queue.isEmpty()) break
        ans++
    }
    return ans
}

//fun maxDistance(s: String, k: Int): Int {
//    class Status(
//        var x: Int,
//        var y: Int,
//        var step: Int,
//    )
//    val direct = mutableMapOf(
//        'N' to intArrayOf(-1, 0),
//        'W' to intArrayOf(0, -1),
//        'S' to intArrayOf(1, 0),
//        'E' to intArrayOf(0, 1),
//    )
//
//    var originX = 0
//    var originY = 0
//    val counts = mutableMapOf<Char, Int>()
//
//    for (c in s) {
//        val (dx, dy) = direct[c] ?: continue
//        originX += dx
//        originY += dy
//        counts[c] = counts.getOrDefault(c, 0) + 1
//    }
//    val originStatus = Status(
//        x = originX,
//        y = originY,
//    )
//
//    val queue = ArrayDeque<Status>()
//    queue.add(originStatus)
//    val visited = mutableSetOf<Pair<Int, Int>>()
//
//    var ans = originStatus.value
//    var cnt = 0
//    while (queue.isNotEmpty() && cnt < k) {
//        cnt++
//        repeat(queue.size) {
//            val status = queue.removeFirst()
//            val newStatusList = listOf(
//                status.copy(x = status.x - direct['N']!![0], y = direct['N']!![1]),
//                status.copy(x = status.x - direct['S']!![0], y = direct['S']!![1]),
//                status.copy(x = status.x - direct['W']!![0], y = direct['W']!![1]),
//                status.copy(x = status.x - direct['E']!![0], y = direct['E']!![1])
//            )
//            for ((dir, delta) in direct) {
//                for (newStatus in newStatusList) {
//                    newStatus.x += delta[0]
//                    newStatus.y += delta[1]
//                    if (newStatus.value <= ans) continue
//                    queue.addLast(newStatus)
//                    ans = maxOf(ans, newStatus.value)
//                }
//            }
//        }
//    }
//    return ans
//}

fun main() {
    println(
        numRollsToTarget(30, 30, 500)
    )
}