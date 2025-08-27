package local

import java.lang.Integer.min
import kotlin.math.max
import kotlin.text.iterator

fun String.to2DIntArray(): Array<IntArray> {
    return this
        .removePrefix("[").removeSuffix("]")
        .split("],[").map {
            it.replace("[", "").replace("]", "")
                .split(",").map(String::toInt).toIntArray()
        }.toTypedArray()
}

fun main() {
    val grid: Array<CharArray> = arrayOf(
        charArrayOf('1', '0', '1', '0', '0'),
        charArrayOf('1', '0', '1', '1', '1'),
        charArrayOf('1', '1', '1', '1', '1'),
        charArrayOf('1', '0', '0', '1', '0')
    )
    println(
        maximalRectangle1(grid)
    )
}

fun maximalRectangle1(matrix: Array<CharArray>): Int {
    val m = matrix.size
    val n = matrix[0].size
    val pair0 = intArrayOf(0, 0)
    val pair1 = intArrayOf(1, 1)
    val dp = Array(m) { Array(n) { IntArray(2) } }

    dp[0][0] = if (matrix[0][0] == '1') pair1 else pair0
    for (i in 1 until m) dp[i][0] = if (matrix[i][0] == '1') {
        intArrayOf(dp[i - 1][0][0] + 1, 1)
    } else pair0
    for (j in 1 until n) dp[0][j] = if (matrix[0][j] == '1') {
        intArrayOf(1, dp[0][j - 1][1] + 1)
    } else pair0

    for (i in 1 until m) {
        for (j in 1 until n) {
            if (matrix[i][j] == '0') {
                dp[i][j] = pair0
                continue
            }
            var x = minOf(dp[i - 1][j - 1][0], dp[i][j - 1][0], dp[i - 1][j][0]) + 1
            var y = minOf(dp[i - 1][j - 1][1], dp[i][j - 1][1], dp[i - 1][j][1]) + 1

            val x2 = min(dp[i][j - 1][0], dp[i - 1][j][0] + 1)
            val y2 = dp[i][j - 1][1] + 1

            val x3 = dp[i - 1][j][0] + 1
            val y3 = min(dp[i - 1][j][1], dp[i][j - 1][1] + 1)

            if (x * y < x2 * y2) {
                x = x2
                y = y2
            }
            if (x * y < x3 * y3) {
                x = x3
                y = y3
            }

            dp[i][j] = intArrayOf(x, y)
        }
    }

    var maxValue = -1
    for (i in 0 until m) {
        for (j in 0 until n) {
            maxValue = max(maxValue, dp[i][j][0] * dp[i][j][1])
        }
    }
  //  println(dp.joinToString("\n") { it.map { it.joinToString("-") }.toString() })
    return maxValue
}

fun myAtoi(s: String): Int {
    val trim = s.trim()
    if (trim.isEmpty()) return 0
    val (prefix, str) = when (trim[0]) {
        '-' -> -1 to trim.substring(1)
        '+' -> 1 to trim.substring(1)
        in '0'..'9' -> 1 to trim
        else -> return 0
    }

    var num = 0.0
    var i = 0
    while (i < str.length) {
        val c = str[i]
        if (c !in '0'..'9') break
        num = num * 10.0 + (c - '0')
        i++
    }
    return (prefix * num).coerceIn(Int.MIN_VALUE.toDouble(), Int.MAX_VALUE.toDouble()).toInt()
}

fun canReach(s: String, minJump: Int, maxJump: Int): Boolean {
    if (s.last() != '0') return false
    val n = s.length
    val lastSeen = IntArray(n)
    val nextSeen = IntArray(n)

    for (i in 0 until n) {
        lastSeen[i] = if (s[i] == '0') i else lastSeen.getOrNull(i - 1) ?: -1
    }
    for (i in (n - 1) downTo 0) {
        nextSeen[i] = if (s[i] == '0') i else nextSeen.getOrNull(i + 1) ?: n
    }
    println((0 until n).toList())
    println(s.toList())
    println(lastSeen.toList())
    println(nextSeen.toList())
    var left = 0
    var right = 0
    while (right < n && left < n) {
        left = left + minJump
        right = min(right + maxJump, n - 1)

        val endIndex = lastSeen[right]
        val startIndex = nextSeen[left]

        if (startIndex > left - minJump + maxJump && endIndex < n - 1) {
            return false
        }


        println("($left-$right): ($startIndex $endIndex)")
        if (startIndex > right || endIndex < left || endIndex < startIndex) {
            return false
        }
        if (endIndex == n - 1) return true
        left = startIndex
        right = endIndex
        if (left + minJump >= n) return false
        while (right + minJump >= n && right > 0) {
            right = lastSeen[right - 1]
        }
        while (left <= right && left < n - 1 && nextSeen[left + minJump] > left + maxJump) {
            left = nextSeen[left + 1]
        }

        if (left > right) return false

    }
    return false
}

fun mincostTickets(days: IntArray, costs: IntArray): Int {
    val n = days.size
    val dp = IntArray(366) { costs[0] }
    dp[n] = costs[0]
    for (i in n downTo 1) {
        val day = days[i - 1]
        val d1 = (day - 1).coerceAtLeast(0)
        val d7 = (day - 6).coerceAtLeast(0)
        val d30 = (day - 29).coerceAtLeast(0)

        for (j in d1..day) {
            dp[j] = min(dp[j], dp[day] + costs[0])
        }
        for (j in d7..day) {
            dp[j] = min(dp[j], dp[day] + costs[1])
        }
        for (j in d30..day) {
            dp[j] = min(dp[j], dp[day] + costs[2])
        }
    }
    println(dp.toList())
    return dp[1]
}

fun numTilings(n: Int): Int {
    when (n) {
        0 -> return 0
        1 -> return 1
        2 -> return 2
        3 -> return 5
    }
    val base = 1000000007
    val dp = LongArray(n + 1)
    dp[0] = 0
    dp[1] = 1
    dp[2] = 2
    dp[3] = 5

    for (i in 4..n) {
        dp[i] = (dp[i - 1] % base) + (dp[i - 2] % base) + (4 * dp[i - 3] % base)
    }
    println(dp.toList())
    return (dp[n] % base).toInt()
}


fun numberOfSpecialChars(word: String): Int {
    val lowerCases = mutableSetOf<Char>()
    val removes = mutableSetOf<Char>()
    val specials = mutableSetOf<Char>()

    for (c in word) {
        if (c in 'a'..'z') {
            if (c !in specials) lowerCases.add(c) else removes.add(c)
        } else if (c in 'A'..'Z') {
            val low = c.lowercaseChar()
            if (low in lowerCases) {
                if (low !in specials) specials.add(low)
            } else removes.add(low)
        }
    }
    println(specials)
    println(removes)
    specials.removeAll(removes)
    return specials.size
}

fun middleNode(head: ListNode?): ListNode? {
    var first = head
    var second = head
    if (head == null) return null
    if (second.next == null) return first

    while (second?.next != null) {
        first = first?.next
        second = second.next?.next
    }

    return first
}

fun strStr(haystack: String, needle: String): Int {
    val n = haystack.length
    val k = needle.length
    if (k > n) return -1

    if (k == n) return if (haystack == needle) 0 else -1

    for (i in 0..(n - k)) {
        val sub = haystack.substring(i, i + k)
        if (sub == needle) {
            return i
        }
    }
    return -1
}

fun isPalindrome(x: Int): Boolean {
    if (x < 0) return false
    if (x < 10) return true
    val digits = mutableListOf<Int>()
    var n = x
    while (n > 0) {
        digits.add(n % 10)
        n /= 10
    }

    var left = 0
    var right = digits.size - 1
    while (left <= right) {
        if (digits[left] != digits[right]) return false
        left++
        right--
    }

    return true
}

fun numberOfWeakCharacters(properties: Array<IntArray>): Int {
    val n = properties.size
    if (n < 2) return 0
    val comparator = compareBy<IntArray>({ -it[0] }, { it[1] })
    val list = properties.sortedWith(comparator)

    var maxValue = 0
    var count = 0
    for (i in 0 until n) {
        val defense = list[i][1]
        if (defense > maxValue) {
            maxValue = defense
        } else count++

    }
    return count
}

fun shortestCommonSupersequence(text1: String, text2: String): String {
    if (text1.isEmpty() || text2.isEmpty()) return text1 + text2
    val m = text1.length
    val n = text2.length

    val dp = Array(m + 1) { IntArray(n + 1) }
    for (i in 0..m) dp[i][0] = 0
    for (j in 0..n) dp[0][j] = 0

    val firstIndexes = mutableSetOf<Int>()
    val secondIndexes = mutableSetOf<Int>()

    for (i in 1..m) {
        for (j in 1..n) {
            dp[i][j] = if (text1[i - 1] == text2[j - 1]) {
                dp[i - 1][j - 1] + 1
            } else {
                max(dp[i - 1][j], dp[i][j - 1])
            }
        }
    }
    var i = m
    var j = n
    val indexPairs = mutableSetOf<Pair<Int, Int>>()
    while (i > 0 && j > 0) {
        if (text1[i - 1] == text2[j - 1]) {
            i--
            j--
            firstIndexes.add(i)
            secondIndexes.add(j)
            indexPairs.add(i to j)
            continue
        }
        if (dp[i][j] == dp[i - 1][j]) i--
        if (dp[i][j] == dp[i][j - 1]) j--
    }
    i = 0
    j = 0
    val shortestSuperString = StringBuilder()
    while (i < m || j < n) {
        val pair = i to j
        if (pair in indexPairs && i < m) {
            shortestSuperString.append(text1[i])
            indexPairs.remove(pair)
            firstIndexes.remove(i)
            secondIndexes.remove(j)
            i++
            j++
            continue
        }
        if (i !in firstIndexes && i < m) {
            shortestSuperString.append(text1[i])
            i++
        }
        if (j !in secondIndexes && j < n) {
            shortestSuperString.append(text2[j])
            j++
        }

    }
    return shortestSuperString.toString()
}

fun longestCommonSubsequence(text1: String, text2: String): Int {
    if (text1.isEmpty() || text2.isEmpty()) return 0
    val m = text1.length
    val n = text2.length

    val dp = Array(m + 1) { IntArray(n + 1) }
    for (i in 0..m) dp[i][0] = 0
    for (j in 0..n) dp[0][j] = 0

    for (i in 1..m) {
        for (j in 1..n) {
            dp[i][j] = if (text1[i - 1] == text2[j - 1]) {
                dp[i - 1][j - 1] + 1
            } else {
                max(dp[i - 1][j], dp[i][j - 1])
            }
        }
    }

    return dp[m][n]
}

fun longestObstacleCourseAtEachPosition(obstacles: IntArray): IntArray {
    val n = obstacles.size
    if (n == 0) return intArrayOf()

    val numbers = obstacles.mapIndexed { index, num ->
        num + index * 1e-8
    }
    val longestList = mutableListOf<Double>()
    val dp = IntArray(n) { 1 }

    for (i in numbers.indices) {
        val num = numbers[i]
        val indexOfNum = longestList.binarySearch(num)
        val isNumInList = indexOfNum >= 0
        if (isNumInList) {
            if (indexOfNum == longestList.lastIndex) {
                longestList.add(indexOfNum + 1, num)
                dp[i] = longestList.size
            } else {
                dp[i] = dp[indexOfNum] + 1
            }
            //   println(longestList)
            continue
        }
        val insertionIndex = -indexOfNum - 1
        if (insertionIndex == longestList.size) {
            longestList.add(num)
            dp[i] = longestList.size
        } else {
            longestList[insertionIndex] = num
            dp[i] = insertionIndex + 1
        }
        //  println(longestList)
    }
    return dp
}

fun maxEnvelopes(envelopes: Array<IntArray>): Int {
    val n = envelopes.size
    if (n == 0) return 0

    val comparator: Comparator<IntArray> = compareBy(
        { it[0] }, { -it[1] }
    )
    val sortedList = envelopes.sortedWith(
        compareBy(
            { it[0] }, { -it[1] }
        ))
    val targetList = mutableListOf<Int>()

    for ((_, h) in sortedList) {
        val idx = targetList.binarySearch(h)
        if (idx >= 0) continue
        val index = -idx - 1
        if (index == targetList.size) {
            targetList.add(h)
        } else {
            targetList[index] = h
        }
    }
    //  println(sortedList.joinToString { it.toList().toString() })
    //  println(targetList.joinToString { it.toString() })
    return targetList.size
}

fun maxEnvelopes2(envelopes: Array<IntArray>): Int {
    val n = envelopes.size
    if (n == 0) return 0

    val list = envelopes.sortedWith(compareBy({ it[0] }, { it[1] }))

    //   println(list.joinToString { it.toList().toString() })
    val dp = IntArray(n) { 1 }
    dp[0] = 1

    var maxValue = 1
    for (i in 1 until n) {
        for (j in 0 until i) {
            if (list[j][0] < list[i][0] && list[j][1] < list[i][1]) {
                dp[i] = max(dp[j] + 1, dp[i])
            }
        }
        maxValue = max(maxValue, dp[i])
    }
    //   println(dp.toList())
    return maxValue
}

fun longestArithSeqLength(nums: IntArray): Int {
    val n = nums.size
    if (n == 0) return 0

    val dp = Array(n) { IntArray(10001) { 1 } }
    dp[0][0] = 1

    var longestLength = 1
    for (i in 1 until n) {
        for (j in 0 until i) {
            val diff = nums[i] - nums[j] + 500
            dp[i][diff] = max(dp[j][diff] + 1, dp[i][diff])
            longestLength = max(longestLength, dp[i][diff])
        }
    }
    //   println(dp.toList())
    return longestLength
}

fun longestArithSeqLength2(nums: IntArray): Int {
    val n = nums.size
    if (n == 0) return 0

    var longestLength = 1

    for (difference in -500..500) {
        val dp = mutableMapOf<Int, Int>()
        for (value in nums) {
            val prev = value - difference
            val currentLength = (dp[prev] ?: 0) + 1
            dp[value] = currentLength
            if (currentLength > longestLength) {
                longestLength = currentLength
            }
        }
    }

    return longestLength
}

fun longestSubsequence(arr: IntArray, difference: Int): Int {
    val n = arr.size
    if (n == 0) return 0

    val dp = mutableMapOf<Int, Int>()
    var longestLength = 1

    for (value in arr) {
        val prev = value - difference
        val currentLength = (dp[prev] ?: 0) + 1
        dp[value] = currentLength
        if (currentLength > longestLength) {
            longestLength = currentLength
        }
    }

    return longestLength
}

fun findLongestChain(pairs: Array<IntArray>): Int {
    val n = pairs.size
    if (n == 0) return 0

    val sortedPairs = pairs.sortedBy { it[0] }


    val dp = IntArray(n) { 1 }
    dp[0] = 1

    var maxValue = 1
    for (i in 1 until n) {
        for (j in 0 until i) {
            if (sortedPairs[j][1] < sortedPairs[i][0]) {
                dp[i] = max(dp[j] + 1, dp[i])
            }
        }
        maxValue = max(maxValue, dp[i])
    }
    // println(dp.toList())
    return maxValue
}


fun findNumberOfLIS(nums: IntArray): Int {
    val n = nums.size
    if (n == 0) return 0

    val dp = IntArray(n) { 1 }
    val count = IntArray(n) { 1 }
    dp[0] = 1
    count[0] = 1

    var maxValue = 1
    var totalCount = 0
    for (i in 0 until n) {
        for (j in i + 1 until n) {
            if (nums[i] < nums[j] && dp[i] + 1 >= dp[j]) {
                if (dp[i] + 1 == dp[j]) {
                    count[j] += count[i]
                } else {
                    count[j] = count[i]
                }
                dp[j] = max(dp[j], dp[i] + 1)
            }
        }
        maxValue = max(maxValue, dp[i])
    }
    for (i in 0 until n) {
        if (dp[i] == maxValue) {
            totalCount += count[i]
        }
    }

    //  println(dp.toList())
    //   println(count.toList())

    return totalCount
}

fun minInsertions(s: String): Int {
    if (s.isEmpty()) return 0
    val text1 = s
    val text2 = s.reversed()
    val m = text1.length
    val n = text2.length

    val dp = Array(m + 1) { IntArray(n + 1) }
    for (i in 0..m) dp[i][0] = 0
    for (j in 0..n) dp[0][j] = 0

    for (i in 1..m) {
        for (j in 1..n) {
            dp[i][j] = if (text1[i - 1] == text2[j - 1]) {
                dp[i - 1][j - 1] + 1
            } else {
                max(dp[i - 1][j], dp[i][j - 1])
            }
        }
    }

    return s.length - dp[m][n]
}

fun maxUncrossedLines(nums1: IntArray, num2: IntArray): Int {
    val m = nums1.size
    val n = num2.size

    if (m == 0 || n == 0) return 0

    val dp = Array(m + 1) { IntArray(n + 1) }
    for (i in 0..m) dp[i][0] = 0
    for (j in 0..n) dp[0][j] = 0

    for (i in 1..m) {
        for (j in 1..n) {
            dp[i][j] = if (nums1[i - 1] == num2[j - 1]) {
                dp[i - 1][j - 1] + 1
            } else {
                max(dp[i - 1][j], dp[i][j - 1])
            }
        }
    }

    return dp[m][n]
}

fun lengthOfLIS(nums: IntArray): Int {
    val n = nums.size
    if (n == 0) return 0

    val longestList = mutableListOf<Int>()

    for (num in nums) {
        val indexOfNum = longestList.binarySearch(num)
        val isNumInList = indexOfNum >= 0
        if (isNumInList) continue
        val insertionIndex = -indexOfNum - 1
        if (insertionIndex == longestList.size) {
            longestList.add(num)
        } else {
            longestList[insertionIndex] = num
        }
    }
    return longestList.size
}


fun lengthOfLIS2(nums: IntArray): Int {
    val n = nums.size
    if (n == 0) return 0

    val dp = IntArray(n) { 1 }

    var maxValue = 1
    for (i in 1 until n) {
        for (j in 0 until i) {
            if (nums[j] < nums[i] && dp[j] + 1 > dp[i]) {
                dp[i] = dp[j] + 1
            }
        }
        maxValue = max(maxValue, dp[i])
    }
    // println(dp.toList())
    return maxValue
}

fun numDistinct(s: String, t: String): Int {
    val m = s.length
    val n = t.length

    val dp = Array(m + 1) { IntArray(n + 1) }
    val count = Array(m + 1) { IntArray(n + 1) { 0 } }
    for (i in 0..m) dp[i][0] = 0
    for (j in 0..n) dp[0][j] = 0

    for (i in 0..m) count[i][0] = 1
    for (j in 0..n) count[0][j] = 1

    for (i in 1..m) {
        for (j in 1..n) {
            dp[i][j] = if (s[i - 1] == t[j - 1]) {
                dp[i - 1][j - 1] + 1
            } else {
                max(dp[i - 1][j], dp[i][j - 1])
            }

            if (s[i - 1] == t[j - 1]) {
                count[i][j] += count[i - 1][j - 1]
            }
            if (dp[i][j] == dp[i - 1][j]) {
                count[i][j] += count[i - 1][j]
            }
            if (dp[i][j] == dp[i][j - 1]) {
                count[i][j] += count[i][j - 1]
            }
        }
    }
    //   println(dp.joinToString("\n") { it.toList().toString() })
    //  println(count.joinToString("\n") { it.toList().toString() })

    return if (dp[m][n] == t.length) count[m][n] else 0
}


fun longestPalindromeSubseq(s: String): Int {
    if (s.isEmpty()) return 0
    val text1 = s
    val text2 = s.reversed()
    val m = text1.length
    val n = text2.length

    val dp = Array(m + 1) { IntArray(n + 1) }
    for (i in 0..m) dp[i][0] = 0
    for (j in 0..n) dp[0][j] = 0

    for (i in 1..m) {
        for (j in 1..n) {
            dp[i][j] = if (text1[i - 1] == text2[j - 1]) {
                dp[i - 1][j - 1] + 1
            } else {
                max(dp[i - 1][j], dp[i][j - 1])
            }
        }
    }

    return dp[m][n]
}

fun wordBreak1(s: String, wordDict: List<String>): Boolean {
    if (s.isEmpty() && wordDict.isEmpty()) return true
    if (s.isEmpty() || wordDict.isEmpty()) return false
    val n = s.length
    val dp = BooleanArray(n + 1)
    dp[0] = true

    for (i in 1..n) {
        dp[i] = false
        val isEndWithWord = wordDict.any { word ->
            i >= word.length && dp[i - word.length] &&
                    s.substring(i - word.length, i) == word
        }
        dp[i] = dp[i] || isEndWithWord
    }
    //  println(dp.toList())
    return dp[n]
}

fun minFallingPathSum(matrix: Array<IntArray>): Int {
    if (matrix.isEmpty()) return 0
    val m = matrix.size
    val n = m
    if (n == 0) return 0

    val dp = Array(m) {
        IntArray(n)
    }
    dp[0][0] = matrix[0][0]

    for (j in 1 until n) dp[0][j] = matrix[0][j]

    var minValue = Int.MAX_VALUE
    for (i in 1 until m) {
        for (j in 0 until n) {
            var min = dp[i - 1][j]
            if (j > 0) min = min(min, dp[i - 1][j - 1])
            if (j < n - 1) min = min(min, dp[i - 1][j + 1])
            dp[i][j] = matrix[i][j] + min
            if (i == m - 1) minValue = min(minValue, dp[i][j])
        }
    }
    return minValue
}