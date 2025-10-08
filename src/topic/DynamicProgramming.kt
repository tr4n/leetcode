package topic

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

fun kConcatenationMaxSum2(arr: IntArray, k: Int): Int {
    val n = arr.size
    val mod = 1_000_000_007
    if (arr.all { it <= 0 }) return 0
    val totalSum = arr.sum().toLong()
    var maxSum = 0L
    var sum = 0L
    for (num in arr) {
        sum = maxOf(sum + num, num.toLong())
        maxSum = maxOf(maxSum, sum)
    }
    if (k > 1) {
        for (num in arr) {
            sum = maxOf(sum + num, num.toLong())
            maxSum = maxOf(maxSum, sum)
        }
    }

    if (k > 2) {
        maxSum = maxOf(maxSum, maxSum + (k - 2) * totalSum, totalSum * k)
    }
    return (maxSum % mod).toInt()
}

fun minimumOperations(grid: Array<IntArray>): Int {
    val m = grid.size
    val n = grid[0].size

    val columnFrequencies = Array(n) { IntArray(11) }
    for (col in 0 until n) {
        for (row in 0 until m) {
            val num = grid[row][col]
            columnFrequencies[col][num]++
        }
    }

    val colChangeCosts = Array(n) { col ->
        IntArray(11) { value ->
            m - columnFrequencies[col][value]
        }
    }

    val minOpsDp = Array(n) { IntArray(11) { m * n } }

    for (value in 0..10) {
        minOpsDp[0][value] = colChangeCosts[0][value]
    }

    for (col in 1 until n) {
        for (currentVal in 0..10) {
            for (prevVal in 0..10) {
                if (prevVal == currentVal) continue
                minOpsDp[col][currentVal] = minOf(
                    minOpsDp[col][currentVal],
                    minOpsDp[col - 1][prevVal] + colChangeCosts[col][currentVal]
                )
            }
        }
    }

    return minOpsDp[n - 1].min()
}

fun numberOfArithmeticSlices(nums: IntArray): Int {
    val n = nums.size
    val numbers = mutableListOf<Long>()
    val numToIndexes = mutableMapOf<Long, MutableList<Int>>()

    for (i in 0 until n) {
        val num = nums[i].toLong()
        numbers.add(num)
        numToIndexes.computeIfAbsent(num) { mutableListOf() }.add(i)
    }

    val dp = Array(n) { IntArray(n) }


    for (i in 2 until n) {
        for (j in 1 until i) {
            val numK = 2L * numbers[j] - numbers[i]
            val indexes = numToIndexes[numK] ?: continue
            for (k in indexes) {
                if (k >= j) break
                dp[i][j] += dp[j][k] + 1
            }
        }
    }
    var total = 0
    for (i in 1 until n) {
        for (j in 0 until i) {
            total += dp[i][j]
        }
    }
    return total
}

fun countPalindromes(s: String): Int {
    val n = s.length
    val indexes = Array(10) { mutableListOf<Int>() }
    for (i in s.indices) {
        val c = s[i]
        indexes[c - '0'].add(i)
    }

    var ans = 0L
    val mod = 1_000_000_007

    val prefix = LongArray(10)
    val prefixPairs = Array(n + 1) { Array(10) { LongArray(10) } }
    for (i in 0 until n) {
        val digit = s[i] - '0'

        for (a in 0 until 10) {
            for (b in 0 until 10) {
                prefixPairs[i + 1][a][b] = prefixPairs[i][a][b]
            }
        }
        for (a in 0 until 10) {
            prefixPairs[i + 1][a][digit] += prefix[a]
            prefixPairs[i + 1][a][digit] %= mod
        }
        prefix[digit]++
    }

    val suffix = LongArray(10)
    val suffixPairs = Array(n + 1) { Array(10) { LongArray(10) } }
    for (i in n - 1 downTo 0) {
        val digit = s[i] - '0'

        for (a in 0 until 10) {
            for (b in 0 until 10) {
                suffixPairs[i][a][b] = suffixPairs[i + 1][a][b]
            }
        }
        for (a in 0 until 10) {
            suffixPairs[i][digit][a] += suffix[a]
            suffixPairs[i][digit][a] %= mod
        }
        suffix[digit]++
    }

    for (mid in 2 until n - 2) {
        val c = s[mid] - '0'
        for (a in 0 until 10) {
            for (b in 0 until 10) {
                val prefixCount = prefixPairs[mid][a][b]
                val suffixCount = suffixPairs[mid + 1][b][a]
                val total = (prefixCount * suffixCount) % mod
                ans = (ans + total) % mod
            }
        }
    }

    return ans.toInt()
}

fun knightProbability(n: Int, k: Int, row: Int, column: Int): Double {
    val dx = intArrayOf(2, 2, -2, -2, 1, 1, -1, -1)
    val dy = intArrayOf(1, -1, 1, -1, 2, -2, 2, -2)
    val dp = Array(k + 1) { Array(n) { DoubleArray(n) } }
    dp[0][row][column] = 1.0

    for (step in 1..k) {
        for (r in 0 until n) {
            for (c in 0 until n) {
                if (dp[step - 1][r][c] == 0.0) continue
                for (i in 0 until 8) {
                    val x = r + dx[i]
                    val y = c + dy[i]
                    if (x !in 0 until n || y !in 0 until n) continue
                    dp[step][x][y] += dp[step - 1][r][c] / 8.0
                }
            }
        }
    }

    var total = 0.0
    for (i in 0 until n) {
        for (j in 0 until n) {
            total += dp[k][i][j]
        }
    }

    return total
}

fun maxProfitAssignment(difficulties: IntArray, profits: IntArray, workers: IntArray): Int {
    workers.sort()
    val n = difficulties.size

    val pairs = (0 until n).map {
        difficulties[it] to profits[it]
    }.sortedWith(compareBy { it.first })

    var maxProfitSofar = Int.MIN_VALUE
    val maxProfits = IntArray(n)

    for (i in 0 until n) {
        val profit = pairs[i].second
        maxProfitSofar = maxOf(maxProfitSofar, profit)
        maxProfits[i] = maxProfitSofar
    }

    var ans = 0
    var last = -1
    for (worker in workers) {
        var l = maxOf(last, 0)
        var r = n - 1
        var best = last

        while (l <= r) {
            val mid = (l + r) / 2
            val diff = pairs[mid].first
            if (diff <= worker) {
                l = mid + 1
                best = mid
            } else {
                r = mid - 1
            }
        }
        if (best < 0) continue
        last = best
        ans += maxProfits[best]
    }
    return ans
}

fun kConcatenationMaxSum(arr: IntArray, k: Int): Int {
    val n = arr.size
    val mod = 1_000_000_007
    if (arr.all { it <= 0 }) return 0
    val totalSum = arr.sum().toLong()
    var maxSum = totalSum * k
    var subSub = 0L
    var sum = 0L
    for (i in 0 until n) {
        sum = maxOf(sum + arr[i], arr[i].toLong())
        subSub = maxOf(subSub, sum)
    }

    maxSum = maxOf(totalSum * (k - 1) + subSub, subSub, maxSum)
    return (maxSum % mod).toInt()
}

fun sumDigitDifferences(nums: IntArray): Long {
    var size = nums[0].toString().length
    var ans = 0L
    var counts = IntArray(10)
    while(size -- > 0) {
        counts = IntArray(10)
        for(i in 0 until  nums.size) {
            val digit = nums[i] % 10
            nums[i] = nums[i]/10
            counts[digit]++
        }
        val list = counts.filter {it > 0}
        for(i in 0 until list.size -1) {
            val a = list[i].toLong()
            for(j in i + 1 until list.size) {
                ans += a * list[j].toLong()
            }
        }
    }
    return ans
}

fun main() {
    println(
        knightProbability(3, 2, 0, 0)
    )
}