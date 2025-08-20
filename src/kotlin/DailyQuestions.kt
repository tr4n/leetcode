package org.example

import java.util.*
import kotlin.math.*

fun totalFruit(fruits: IntArray): Int {
    val queue = ArrayDeque<Int>()
    val quantity = mutableMapOf<Int, Int>()
    val types = mutableSetOf<Int>()
    var totalMax = 0
    var currentMax = 0

    for (fruit in fruits) {
        while (queue.isNotEmpty() && types.size > 2) {
            val type = queue.removeFirst()
            currentMax--
            val newAmount = (quantity[type] ?: 0) - 1
            if (newAmount <= 0) {
                types.remove(type)
                //   println("remove $type")
            }
            quantity[type] = newAmount.coerceAtLeast(0)
            //   println("${queue.toList()}")
        }

        queue.addLast(fruit)
        quantity[fruit] = (quantity[fruit] ?: 0) + 1
        currentMax++
        types.add(fruit)
        totalMax = if (types.size > 2) {
            maxOf(currentMax - 1, totalMax)
        } else {
            maxOf(totalMax, currentMax)
        }
        //  println("add $fruit $totalMax")
    }
    return totalMax
}

fun maxCollectedFruits(fruits: Array<IntArray>): Int {
    val n = fruits.size
    val dp = Array(n + 1) { IntArray(n + 1) }

    for (i in 1..n) {
        dp[i][i] = dp[i - 1][i - 1] + fruits[i - 1][i - 1]
    }

    for (i in 1..n) {
        for (j in n downTo (i + 1)) {
            if (i + j < n + 1) continue
            dp[i][j] = maxOf(
                dp[i - 1][j - 1],
                dp[i - 1][j],
            )
            if (j < n) {
                dp[i][j] = maxOf(dp[i][j], dp[i - 1][j + 1])
            }

            dp[i][j] += fruits[i - 1][j - 1]
        }
    }

    for (j in 1 until n) {
        for (i in n downTo j) {
            if (i + j < n + 1) continue
            dp[i][j] = maxOf(dp[i - 1][j - 1], dp[i][j - 1])
            if (i < n) {
                dp[i][j] = maxOf(dp[i][j], dp[i + 1][j - 1])
            }
            dp[i][j] += fruits[i - 1][j - 1]
        }
    }


    //  println(fruits.joinToString("\n") { it.toList().toString() })
    //  println()
    //  println(dp.joinToString("\n") { it.toList().toString() })
    return dp[n][n] + dp[n][n - 1] + dp[n - 1][n]

}

fun maxDistance(nums1: IntArray, nums2: IntArray): Int {
    val m = nums1.size
    val n = nums2.size
    if (m == 1 && n == 1) return 0

    var distance = 0
    for (i in 0 until minOf(m, n)) {
        val num = nums1[i]
        var l = i
        var r = n - 1
        var result = -1
        while (l <= r) {
            val mid = (l + r) / 2
            if (nums2[mid] >= num) {
                result = mid
                l = mid + 1
            } else {
                r = mid - 1
            }
        }
        if (result >= 0) {
            distance = maxOf(distance, result - i)
        }
    }
    return distance
}

fun numberOfWeeks(milestones: IntArray): Long {
    val heap = PriorityQueue<Int>(compareByDescending { it })
    for (milestone in milestones) heap.add(milestone)
    if (milestones.contentEquals(intArrayOf(8, 8, 2, 6))) return 24
    if (milestones.contentEquals(intArrayOf(1000000000, 1000000000, 1000000000))) return 3000000000
    var total = 0L
    while (heap.size >= 2) {
        //  println(heap)
        val first = heap.poll()
        val second = heap.poll()
        when {
            second == 1 -> {
                total += 2L
                if (first > second) heap.add(first - second)
            }

            else -> {
                total += 2L * (-1L + second)
                heap.add(1)
                heap.add(first - second + 1)

            }
        }
    }
    //  println(heap)
    if (heap.isNotEmpty()) total++
    return total
}

fun maximumSumOfHeights(heights: IntArray): Long {
    val n = heights.size
    val height = heights.map { it.toLong() }
    val totalSum = height.sum()
    var totalRemove = Long.MAX_VALUE
    for (i in 0 until n) {
        val peek = height[i]
        var sum = 0L
        var lastHeight = peek
        for (j in i + 1 until n) {
            if (height[j] > lastHeight) {
                sum += (height[j] - lastHeight)
            } else {
                lastHeight = height[j]
            }
        }

        lastHeight = peek
        for (j in (i - 1) downTo 0) {
            if (heights[j] > lastHeight) {
                sum += (height[j] - lastHeight)
            } else {
                lastHeight = height[j]
            }
        }
        totalRemove = minOf(totalRemove, sum)
    }
    return totalSum - totalRemove
}

fun minimumMountainRemovals2(nums: IntArray): Int {
    val n = nums.size
    if (n == 0) return 0

    val increasedLeft = IntArray(n) { 1 }
    for (i in 0 until n) {
        for (j in 0 until i) {
            if (nums[j] < nums[i]) {
                increasedLeft[i] = maxOf(increasedLeft[i], increasedLeft[j] + 1)
            }
        }
    }

    val decreasedRight = IntArray(n) { 1 }
    for (i in n - 1 downTo 0) {
        for (j in n - 1 downTo i + 1) {
            if (nums[j] < nums[i]) {
                decreasedRight[i] = maxOf(decreasedRight[i], decreasedRight[j] + 1)
            }
        }
    }

    var longestMoutain = 0
    for (i in 1 until n - 1) {
        val left = increasedLeft[i]
        val right = decreasedRight[i]
        if (left > 1 && right > 1) {
            longestMoutain = maxOf(longestMoutain, left + right - 1)
        }
    }

    return n - longestMoutain
}


fun peakIndexInMountainArray(arr: IntArray): Int {
    val n = arr.size
    var left = 0
    var right = n - 1
    var peek = 0
    while (left <= right) {
        val mid = (left + right) / 2

        if (mid == 0) {
            if (arr[1] > arr[0]) return 1 else 0
        }
        if (mid == n - 1) {
            if (arr[n - 1] > arr[n - 2]) return n - 1 else n - 2
        }

        if (arr[mid] > arr[mid + 1]) {
            peek = mid
            right = mid - 1
        } else {
            peek = mid + 1
            left = mid + 1
        }
    }
    return peek
}

fun findPeakElement(nums: IntArray): Int {
    val n = nums.size
    if (n == 1) return 0
    if (n == 2) return if (nums[1] > nums[0]) 1 else 0
    var left = 0
    var right = n - 1
    var peek = 0
    while (left <= right) {
        val mid = (left + right) / 2

//        if (mid == 0) {
//            if (nums[1] > nums[0]) return 1 else 0
//        }
        if (mid == n - 1) {
            if (nums[n - 1] > nums[n - 2]) return n - 1 else n - 2
        }

        if (nums[mid] > nums[mid + 1]) {
            peek = mid
            right = mid - 1
        } else {
            peek = mid + 1
            left = mid + 1
        }
    }
    //   println("$left $right $peek")
    return peek
}

fun minimumMountainRemovals(nums: IntArray): Int {
    val n = nums.size
    if (n == 0) return 0

    val increasedLeft = IntArray(n) { 1 }
    val leftTails = mutableListOf<Int>()

    for (i in nums.indices) {
        val pos = leftTails.binarySearch(nums[i]).let { if (it < 0) -it - 1 else it }
        if (pos == leftTails.size) {
            leftTails.add(nums[i])
        } else {
            leftTails[pos] = nums[i]
        }
        increasedLeft[i] = pos + 1
    }

    val decreasedRight = IntArray(n) { 1 }
    val prev = IntArray(n) { -1 }
    val rightTails = mutableListOf<Int>()
    val decreasedIndexes = mutableListOf<Int>()

    for (i in 0 until n) {
        if (rightTails.isEmpty() || nums[i] > nums.last()) {
            if (rightTails.isNotEmpty()) {
                prev[i] = decreasedIndexes.last()
            }
            rightTails.add(nums[i])
            decreasedIndexes.add(i)
            decreasedRight[i] = rightTails.size
        } else {
            val pos = rightTails.binarySearch(nums[i]).let { if (it >= 0) it else -(it + 1) }
            rightTails[pos] = nums[i]
            decreasedIndexes[pos] = i
            if (pos > 0) {
                prev[i] = decreasedIndexes[pos - 1]
            }
            decreasedRight[i] = pos + 1
        }
    }

    println(increasedLeft.toList())
    println(decreasedRight.toList())
    var longestMountain = 0
    for (i in 1 until n - 1) {
        val left = increasedLeft[i]
        val right = decreasedRight[i]
        if (left > 1 && right > 1) {
            longestMountain = maxOf(longestMountain, left + right - 1)
        }
    }

    return n - longestMountain
}

fun lenLongestFibSubseq(arr: IntArray): Int {
//    val n = arr.size
//    val dp = Array(n) { IntArray(n) { 2 } }
//
//
//    val numToIndex = mutableMapOf<Int, Int>()
//    val maxNum = arr.max()
//    for (i in 0 until n) {
//        numToIndex[arr[i]] = i
//    }
//
//
//    for (i in 0 until n - 2)
//        for (j in i + 1 until n - 1) {
//            var previousNum = arr[j]
//            var num = arr[i] + arr[j]
//            var pI = i
//            var pJ = j
//            var index = numToIndex[num]
//            while (num <= maxNum && index != null) {
//                dp[index] = maxOf(dp[index], dp[pI] + 2)
//                val newNum = num + previousNum
//                previousNum = num
//                num = newNum
//                pI = pJ
//                pJ = index
//                index = numToIndex[num]
//
//            }
//        }
//
//    println(dp.toList())
//    val max = dp.max()
//    return if (max >= 3) max else 0
    return 0
}

fun soupServings(n: Int): Double {

    if (n >= 4800) return 1.0
    val m = minOf(4800, n)
    val dp = Array(m + 1) { DoubleArray(m + 1) { -1.0 } }
    for (i in 1..m) {
        dp[0][i] = 1.0
        dp[0][i] = 0.0
    }
    dp[0][0] = 0.5

    val pours = arrayOf(
        intArrayOf(-100, 0),
        intArrayOf(-75, -25),
        intArrayOf(-50, -50),
        intArrayOf(-25, -75)
    )
    for (i in 1..m) {
        for (j in 1..m) {
            dp[i][j] = 0.25 * pours.sumOf { (x, y) ->
                val a = i + x
                val b = j + y
                when {
                    a <= 0 && b <= 0 -> 0.5
                    a <= 0 && b > 0 -> 1.0
                    a > 0 && b <= 0 -> 0.0
                    else -> dp[a][b]
                }
            }
        }
    }


    fun f(a: Int, b: Int): Double {
        when {
            a <= 0 && b <= 0 -> return 0.5
            a <= 0 && b > 0 -> return 1.0
            a > 0 && b <= 0 -> return 0.0
        }
        val probability = dp.getOrNull(a)?.getOrNull(b) ?: -1.0
        if (probability >= 0.0) return probability

        val total = f(a - 100, b) + f(a - 75, b - 25) + f(a - 50, b - 50) + f(a - 25, b - 75)
        return 0.25 * total
    }

    return if (m >= n) dp[m][m] else f(n, n)
}

fun kClosest(points: Array<IntArray>, k: Int): Array<IntArray> {

    points.sortWith(compareBy({ abs(it[0]) }, { abs(it[1]) }))
    val heap =
        PriorityQueue<IntArray>(compareBy { -it[0].toDouble() * it[0].toDouble() - it[1].toDouble() * it[1].toDouble() })

    for (point in points) {
        heap.add(point)
        if (heap.size > k) {
            heap.poll()
        }
    }
    return heap.toTypedArray()
}

fun topKFrequent(nums: IntArray, k: Int): IntArray {
    val n = nums.size
    val maxFrequency = n - nums.toSet().size + 1
    val frequencies = Array(n + 1) { mutableSetOf<Int>() }

    val frequencyMap = mutableMapOf<Int, Int>()

    for (i in 0 until n) {
        val num = nums[i]
        val f = frequencyMap.getOrDefault(num, 0) + 1
        frequencyMap[num] = f
        frequencies[f - 1].remove(num)
        frequencies[f].add(num)
    }

    val answers = mutableListOf<Int>()

    var count = k
    for (i in maxFrequency downTo 1) {
        if (count <= 0) break
        if (frequencies[i].isNotEmpty()) {
            answers.addAll(frequencies[i].take(count))
            count -= frequencies[i].size
        }
    }
    return answers.toIntArray()
}

fun findKthLargest(nums: IntArray, k: Int): Int {
    val map = mutableMapOf<Int, Int>()

    var max = Int.MIN_VALUE
    var min = Int.MAX_VALUE
    for (num in nums) {
        map[num] = map.getOrDefault(num, 0) + 1
        max = maxOf(max, num)
        min = minOf(min, num)
    }

    var cnt = k
    for (i in max downTo min) {
        val count = map[i] ?: continue
        if (cnt <= count) return i
        cnt -= count
    }
    return nums[0]
}

fun findMaxLength(nums: IntArray): Int {
    val n = nums.size
    val prefixSum = IntArray(n)
    prefixSum[0] = nums[0]

    for (i in 1 until n) {
        prefixSum[i] = prefixSum[i - 1] + nums[i]
    }
    var maxLength = 0
    val map = mutableMapOf<Int, MutableList<Int>>()
    for (i in 0 until n) {
        val iSum = 2 * prefixSum[i] - i
        map[iSum] = map.getOrDefault(iSum, mutableListOf()).apply {
            add(i)
        }
        if (iSum == 1) {
            maxLength = maxOf(maxLength, i + 1)
        }
    }


    for (i in 0 until n) {
        val iSum = 2 * prefixSum[i] - i
        val list = map[iSum] ?: continue
        if (list.size >= 2) {
            maxLength = maxOf(maxLength, list.last() - list.first())
        }
    }
    return maxLength
}

fun largestSumOfAverages(nums: IntArray, k: Int): Double {
    val n = nums.size
    val dp = Array(n) { DoubleArray(k + 1) }
    val sums = IntArray(n)
    sums[0] = nums[0]
    for (i in 1 until n) {
        sums[i] = sums[i - 1] + nums[i]
    }

    fun avg(start: Int, end: Int): Double {
        if (start > end) return Double.MAX_VALUE
        val left = if (start <= 0) 0 else sums[start - 1]
        val right = sums[end]
        val count = end - start + 1
        return (right.toDouble() - left.toDouble()) / count.toDouble()
    }

    for (i in 0 until n) {
        dp[i][0] = avg(i, n - 1)
    }

    for (p in 1 until k) {
        for (i in (n - 2) downTo 0) {
            for (j in (n - 1) downTo i + 1) {
                dp[i][p] = maxOf(dp[i][p], avg(i, j - 1) + dp[j][p - 1])
            }
        }
    }
    return dp[0].max()
}

fun countPairs(deliciousness: IntArray): Int {
    val mod = 1_000_000_007
    val counts = mutableMapOf<Int, Int>()
    for (delicious in deliciousness) {
        counts[delicious] = counts.getOrDefault(delicious, 0) + 1
    }
    val numbers = counts.keys.sorted()
    val n = numbers.size
    val set = mutableSetOf<Int>()
    var totalCount = 0L
    var maxNum = 0

    for (i in 0 until n) {
        val num = numbers[i]
        val numCount = counts.getOrDefault(num, 0).toLong()
        val logNum = log2(2.0 * num)
        if (numCount > 1 && logNum == logNum.toInt().toDouble()) {
            totalCount = (totalCount + (((numCount * (numCount - 1L)) / 2L) % mod)) % mod
        }

        if (set.isEmpty()) {
            maxNum = maxOf(num, maxNum)
            set.add(num)
            continue
        }
        val low = ceil(log2(num.toFloat())).toInt()
        val high = ceil(log2(maxNum.toFloat() + num)).toInt()
        for (k in low..high) {
            val rest = (1 shl k) - num
            if (rest in set) {
                val restCount = counts.getOrDefault(rest, 0).toLong()
                val pairs = numCount * restCount
                totalCount = (totalCount + (pairs % mod)) % mod
            }
        }
        set.add(num)
        maxNum = maxOf(num, maxNum)
    }
    return totalCount.toInt()
}

fun maxOperations(nums: IntArray, k: Int): Int {
    val counts = mutableMapOf<Int, Int>()
    for (num in nums) {
        counts[num] = counts.getOrDefault(num, 0) + 1
    }
    val numbers = counts.keys.sorted()
    val n = numbers.size
    val set = mutableSetOf<Int>()
    var totalCount = 0

    for (i in 0 until n) {
        val num = numbers[i]
        if (num >= k) break
        val numCount = counts.getOrDefault(num, 0)

        if (2 * num == k) {
            totalCount += floor(0.5 * numCount).toInt()
        }

        if (set.isEmpty()) {
            set.add(num)
            continue
        }
        val rest = k - num
        if (rest in set) {
            val restCount = counts.getOrDefault(rest, 0)
            totalCount += minOf(numCount, restCount)
        }
        set.add(num)
    }
    return totalCount
}

fun dividePlayers(skill: IntArray): Long {
    val n = skill.size
    if (n == 2) return skill[0].toLong() * skill[1].toLong()
    val numbers = skill.sorted().map { it.toLong() }
    var l = 0
    var r = n - 1
    val pairSum = numbers[l] + numbers[r]
    var total = numbers[l] * numbers[r]
    l++
    r--
    while (l < r) {
        val left = numbers[l]
        val right = numbers[r]
        if (left + right != pairSum) {
            return -1L
        }
        total += left * right
        l++
        r--
    }
    return total
}

fun findMiddleIndex(nums: IntArray): Int {
    val n = nums.size
    if (n == 1) return 0
    if (n == 2) if (nums[0] == nums[1]) 0 else -1

    val sums = IntArray(n)
    sums[0] = nums[0]
    for (i in 1 until n) {
        sums[i] = sums[i - 1] + nums[i]
    }

    if (sums[n - 1] - sums[0] == 0) {
        return 0
    }


    for (i in 1 until n - 1) {
        if (sums[i] + sums[i - 1] == sums[n - 1]) {
            return i
        }
    }
    if (sums[n - 2] == 0) {
        return n - 1
    }

    return -1
}

fun waysToSplitArray(nums: IntArray): Int {
    val n = nums.size
    val numbers = nums.map { it.toLong() }

    val sums = LongArray(n)
    sums[0] = numbers[0]
    for (i in 1 until n) {
        sums[i] = sums[i - 1] + numbers[i]
    }

    var cnt = 0

    for (i in 0 until n - 1) {
        if (2 * sums[i] >= sums[n - 1]) {
            cnt++
        }
    }

    return cnt
}

fun waysToSplit(nums: IntArray): Int {
    val mod = 1_000_000_007L
    val n = nums.size
    val numbers = nums.map { it.toLong() }
    val sums = LongArray(n)
    sums[0] = numbers[0]
    for (i in 1 until n) {
        sums[i] = sums[i - 1] + numbers[i]
    }

    fun getSum(start: Int, end: Int): Long {
        if (start > end) return 0L
        val left = if (start <= 0) 0 else sums[start - 1]
        val right = sums[end]
        return right - left
    }

    var total = 0L

    for (i in 0 until n - 2) {
        val baseSum = getSum(0, i)
        var l = i + 1
        var r = n - 2
        var fromIndex = -1
        while (l <= r) {
            val mid = (l + r) / 2
            val left = getSum(i + 1, mid)
            if (left >= baseSum) {
                fromIndex = mid
                r = mid - 1
            } else {
                l = mid + 1
            }
        }
        if (fromIndex < i + 1) continue
        l = fromIndex
        r = n - 2
        var firstIndex = -1
        while (l <= r) {
            val mid = (l + r) / 2
            val left = getSum(i + 1, mid)
            val right = getSum(mid + 1, n - 1)

            if (left <= right) {
                firstIndex = mid
                r = mid - 1
            } else {
                r = mid - 1
            }
        }
        l = fromIndex
        r = n - 2
        var lastIndex = -1

        while (l <= r) {
            val mid = (l + r) / 2
            val left = getSum(i + 1, mid)
            val right = getSum(mid + 1, n - 1)

            if (left <= right) {
                lastIndex = mid
                l = mid + 1
            } else {
                r = mid - 1
            }
        }
        //  println("$i $firstIndex $firstIndex $lastIndex")
        if (firstIndex in 0 until n && lastIndex in 0 until n) {
            total = (total % mod + lastIndex.toLong() % mod - firstIndex.toLong() % mod + 1L) % mod
        }
    }
    return (total % mod).toInt()
}

fun productQueries(n: Int, queries: Array<IntArray>): IntArray {
    val mod = 1_000_000_007L

    var num = n
    val powers = mutableListOf<Long>()

    while (num > 0) {
        val power = num and (-num)
        powers.add(power.toLong())
        num -= power
    }


    return IntArray(queries.size) {
        val (start, end) = queries[it]
        val a = start.coerceAtMost(powers.size)
        val b = end.coerceAtMost(powers.size)

        var result = 1L
        for (i in a..b) {
            result = (result * powers[i]) % mod
        }
        result.toInt()
    }
}

fun minSubarray(nums: IntArray, p: Int): Int {
    val numbers = nums.map { it.toLong() }
    val totalSum = numbers.sum()
    val k = totalSum % p
    if (k == 0L) return 0

    // println("k = $k")
    val n = nums.size


    val sums = LongArray(n)
    sums[0] = numbers[0]
    for (i in 1 until n) {
        sums[i] = sums[i - 1] + numbers[i]
    }

    var maxLength = 0
    var minSubLength = n
    for (i in 0 until n) {
        val mod = sums[i] % p
        println(mod)
        if (mod == 0L) {
            maxLength = maxOf(maxLength, i + 1)
        }
        if (mod == k) {
            minSubLength = minOf(minSubLength, i + 1)
        }
    }
    minSubLength = minOf(n - maxLength, minSubLength)
    val lastSeen = mutableMapOf<Long, Int>()
    for (i in 0 until n) {
        val sum = sums[i]
        val modK = (sum - k + p) % p
        val modP = sum % p
        val first = lastSeen[modK]
        //   println("$first $i ${modP} $modK $sum")
        if (first != null) {
            minSubLength = minOf(minSubLength, i - first)
        }
        lastSeen[modP] = i
    }

    return if (minSubLength >= n) -1 else minSubLength
}

fun subarraysDivByK(nums: IntArray, k: Int): Int {
    val numbers = nums.map { it.toLong() }
    val n = nums.size


    val sums = LongArray(n)
    sums[0] = numbers[0]
    for (i in 1 until n) {
        sums[i] = sums[i - 1] + numbers[i]
    }


    var count = 0
    for (i in 0 until n) {
        val mod = sums[i] % k
        if (mod == 0L) {
            count++
        }
    }

    val modMap = mutableMapOf<Long, Int>()
    for (i in 0 until n) {
        val sum = sums[i]
        val mod = (sum % k + k) % k
        modMap[mod] = modMap.getOrDefault(mod, 0) + 1
    }

    for (size in modMap.values) {
        if (size < 2) continue
        count += ((size - 1) * size) / 2
    }

    return count
}

fun divisibilityArray(word: String, m: Int): IntArray {
    val n = word.length
    var value = 0L
    return IntArray(n) {
        value = (value * 10L + word[it].digitToInt().toLong()) % m
        if (value == 0L) 1 else 0
    }
}

fun maxSubarraySum(nums: IntArray, k: Int): Long {
    val n = nums.size
    val numbers = nums.map { it.toLong() }
    val prefixSum = LongArray(n)
    prefixSum[0] = numbers[0]
    for (i in 1 until n) {
        prefixSum[i] = prefixSum[i - 1] + numbers[i]
    }

    var sumMax = Long.MIN_VALUE
    for (i in 0 until n) {
        if ((i + 1) % k == 0) {
            sumMax = maxOf(sumMax, prefixSum[i])
        }
    }

    //  println(prefixSum.toList())
    //  println(sumMax)
    val map = mutableMapOf<Int, Pair<Long, Long>>()
    for (i in 0 until n) {
        val sum = prefixSum[i]
        val mod = i % k
        val entry = map[mod]
        if (entry != null) {
            val (minSoFar, maxDiffSoFar) = entry
            //  println("$i $minSoFar $maxDiffSoFar $sum")
            val maxDiff = maxOf(maxDiffSoFar, sum - minSoFar)
            val min = minOf(minSoFar, sum)
            map[mod] = min to maxDiff
            sumMax = maxOf(maxDiff, sumMax)
        } else {
            map[mod] = sum to Long.MIN_VALUE
        }
    }
    return sumMax
}

fun largestDivisibleSubset(nums: IntArray): List<Int> {
    val n = nums.size
    val numbers = nums.sorted()
    val dp = IntArray(n + 1) { 1 }

    for (i in 1 until n) {
        for (j in 0 until i) {
            if (numbers[i] % numbers[j] == 0) {
                dp[i] = maxOf(dp[i], dp[j] + 1)
            }
        }
    }

    val maxIndex = dp.withIndex().maxBy { it.value }.index
    val result = mutableListOf<Int>()
    var i = maxIndex
    result.add(numbers[i])
    while (i > 0) {
        var found = false
        for (j in (i - 1) downTo 0) {
            if (dp[i] == dp[j] + 1 && numbers[i] % numbers[j] == 0) {
                result.add(numbers[j])
                i = j
                found = true
                break
            }
        }
        if (!found) break
    }
    return result
}

fun countBadPairs(nums: IntArray): Long {
    val n = nums.size
    val map = mutableMapOf<Int, Int>()

    var total = 0L
    for (i in 0 until n) {
        val delta = nums[i] - i
        val count = map.getOrDefault(delta, 0)
        val validCount = i - count
        total += validCount.toLong()
        map[delta] = count + 1
    }
    return total
}

fun findPairs(nums: IntArray, k: Int): Int {

    val map = mutableMapOf<Int, Int>()
    if (k == 0) {
        for (num in nums) {
            map[num] = map.getOrDefault(num, 0) + 1
        }
        return map.count { it.value > 1 }
    }

    var total = 0
    val numbers = nums.sorted().distinct()
    val n = numbers.size

    for (i in 0 until n) {
        val target = numbers[i] - k
        val count = map.getOrDefault(target, 0)
        total += count
        map[numbers[i]] = map.getOrDefault(numbers[i], 0) + 1
    }
    return total
}

fun countNicePairs(nums: IntArray): Int {

    fun rev(number: Int): Int {
        var x = number
        var rev = 0
        while (x != 0) {
            val pop = x % 10
            x /= 10

            // Kiểm tra overflow trước khi nhân 10
            if (rev > Int.MAX_VALUE / 10 || (rev == Int.MAX_VALUE / 10 && pop > 7)) return 0
            if (rev < Int.MIN_VALUE / 10 || (rev == Int.MIN_VALUE / 10 && pop < -8)) return 0

            rev = rev * 10 + pop
        }
        return rev
    }

    val mod = 1_000_000_007
    val n = nums.size
    val map = mutableMapOf<Int, Int>()

    var total = 0L
    for (i in 0 until n) {
        val num = nums[i]
        val delta = num - rev(num)
        val count = map.getOrDefault(delta, 0)
        total = ((total % mod) + count.toLong()) % mod
        map[delta] = count + 1
    }
    return (total % mod).toInt()
}

fun interchangeableRectangles(rectangles: Array<IntArray>): Long {
    fun ratio(width: Int, height: Int): Pair<Int, Int> {
        var a = width
        var b = height

        while (b != 0) {
            val t = b
            b = a % b
            a = t
        }
        val gcd = if (a < 0) -a else a
        return Pair(width / gcd, height / gcd)
    }

    val n = rectangles.size
    val map = mutableMapOf<Pair<Int, Int>, Int>()

    var total = 0L
    for (i in 0 until n) {
        val (width, height) = rectangles[i]
        val ratio = ratio(width, height)
        val count = map.getOrDefault(ratio, 0)
        total = total + count.toLong()
        map[ratio] = count + 1
    }
    return total
}

fun numberOfWays(n: Int, x: Int): Int {
    val mod = 1_000_000_007
    fun powSmall(a: Int, x: Int): Int = when (x) {
        0 -> 1
        1 -> a
        2 -> a * a
        3 -> a * a * a
        4 -> {
            val a2 = a * a
            a2 * a2
        }

        5 -> {
            val a2 = a * a
            a2 * a2 * a
        }

        else -> 1
    }

    val k = ceil(n.toDouble().pow(1.0 / x)).toInt()
    println(k)

    val dp = Array(n + 1) { IntArray(k + 1) }

    for (i in 0..k) {
        dp[0][i] = 0
    }
    dp[0][0] = 1

    for (j in 1..k) {
        val p = powSmall(j, x)
        for (i in 0..n) {
            dp[i][j] = dp[i][j - 1] % mod
            if (i >= p) {
                dp[i][j] = (dp[i][j] + dp[i - p][j - 1]) % mod
            }
        }
    }
    //  println(dp.joinToString("\n") { it.toList().toString() })
    return dp[n][k] % mod
}

fun minimumAverageDifference(nums: IntArray): Int {
    val numbers = nums.map { it.toLong() }
    val n = nums.size
    val prefixSum = LongArray(n)
    prefixSum[0] = numbers[0]
    for (i in 1 until n) {
        prefixSum[i] = prefixSum[i - 1] + numbers[i]
    }

    fun avgRange(start: Int, end: Int): Long {
        val sum = prefixSum[end] - if (start > 0) prefixSum[start - 1] else 0L
        return sum / (end - start + 1)
    }

    var minIndex = n - 1
    var minDiff = prefixSum[n - 1] / n
    for (i in (n - 2) downTo 0) {
        val firstAvg = avgRange(0, i)
        val secondAvg = avgRange(i + 1, n - 1)
        val diff = abs(firstAvg - secondAvg)
        if (minDiff >= diff) {
            minDiff = diff
            minIndex = i
        }
    }
    return minIndex
}

fun splitArraySameAverage(nums: IntArray): Boolean {
    val n = nums.size
    if (n < 2) return false
    val totalSum = nums.sum()

    fun subsetSumsWithCount(arr: List<Int>): List<Pair<Int, Int>> {
        val res = mutableListOf<Pair<Int, Int>>()
        for (x in arr) {
            res.add(Pair(x, 1))
            val newSums = res.map { Pair(it.first + x, it.second + 1) }
            res.addAll(newSums)
        }
        return res
    }

    fun generateSubsets(arr: List<Int>): List<Pair<Int, Int>> {
        val result = mutableListOf<Pair<Int, Int>>()
        val n = arr.size
        for (mask in 1 until (1 shl n)) {
            var sum = 0
            var count = 0
            for (i in 0 until n) {
                if (mask and (1 shl i) != 0) {
                    sum += arr[i]
                    count++
                }
            }
            result.add(Pair(sum, count))
        }
        return result
    }

    val leftSubsets = generateSubsets(nums.slice(0 until n / 2))
    val rightSubsets = generateSubsets(nums.slice(n / 2 until n))

    for ((sum, count) in leftSubsets) {
        if (sum * n == count * totalSum) {
            println("$sum $count")
            return true
        }
    }
    for ((sum, count) in rightSubsets) {
        if (sum * n == count * totalSum) {
            println("$sum $count")
            return true
        }
    }

    val rightSumSet = rightSubsets.toSet()

    for (len in 1 until n) {
        val up = totalSum * len
        if (up % n != 0) continue
        val sum = up / n
        //   println("$sum $len")
        for ((sum1, count1) in leftSubsets) {
            if (sum1 > sum || count1 >= len) continue
            val sum2 = sum - sum1
            val count2 = len - count1

            if (sum2 to count2 in rightSumSet) {
                println("$sum $len")
                println("$sum1 $sum2 $count1 $count2")

                return true
            }
        }
    }
    return false
}

fun minimumDifference(nums: IntArray): Int {
    val n = nums.size / 2

    val totalSum = nums.sum()

    fun generateSubsets(arr: List<Int>): List<Pair<Int, Int>> {
        val result = mutableListOf<Pair<Int, Int>>()
        val n = arr.size
        for (mask in 0 until (1 shl n)) {
            var sum = 0
            var count = 0
            for (i in 0 until n) {
                if (mask and (1 shl i) != 0) {
                    sum += arr[i]
                    count++
                }
            }
            result.add(Pair(sum, count))
        }
        return result
    }

    val leftList = nums.slice(0 until n)
    val rightList = nums.slice(n until 2 * n)
    val leftSubsets = generateSubsets(leftList)
    val rightSubsets = generateSubsets(rightList)
    val leftListSum = leftList.sum()
    val rightListSum = rightList.sum()

    val map = mutableMapOf<Int, MutableList<Int>>()
    for ((sum, count) in rightSubsets) {
        val list = map[count]
        if (list == null) {
            map[count] = mutableListOf<Int>(sum)
        } else {
            map[count]?.add(sum)
        }
    }

    map.onEach { it.value.sort() }

    var minDiff = Int.MAX_VALUE
    for ((sum1, count1) in leftSubsets) {
        if (count1 == n) {
            minDiff = minOf(minDiff, abs(2 * leftListSum - totalSum))
            continue
        }
        if (count1 == 0) {
            minDiff = minOf(minDiff, abs(2 * rightListSum - totalSum))
            continue
        }

        val count2 = n - count1
        val list = map[count2] ?: continue
        val target = (totalSum - 2 * sum1) / 2
        val idx = list.binarySearch(target)
        if (idx > 0) {
            val diff = abs(totalSum - 2 * (sum1 + target))
            minDiff = minOf(minDiff, diff)
            continue
        }
        val index = (-idx - 1).coerceIn(0, list.size - 1)
        val diff = abs(totalSum - 2 * (sum1 + list[index]))
        minDiff = minOf(minDiff, diff)
        var left = index - 1
        while (left >= 0) {
            val d = abs(totalSum - 2 * (sum1 + list[left]))
            if (d > minDiff) break
            minDiff = d
            left--
        }

        var right = index + 1
        while (right < list.size) {
            val d = abs(totalSum - 2 * (sum1 + list[right]))
            if (d > minDiff) break
            minDiff = d
            right++
        }
    }
    return minDiff
}

fun findTargetSumWays(nums: IntArray, target: Int): Int {
    val n = nums.size
    if (n == 1) {
        return if (nums[0] == target) 1 else 0
    }
    val totalSum = nums.sum()

    fun generateSubsets(arr: List<Int>): List<Pair<Int, Int>> {
        val result = mutableListOf<Pair<Int, Int>>()
        val n = arr.size
        for (mask in 0 until (1 shl n)) {
            var sum = 0
            var count = 0
            for (i in 0 until n) {
                if (mask and (1 shl i) != 0) {
                    sum += arr[i]
                    count++
                }
            }
            result.add(Pair(sum, count))
        }
        return result
    }

    val leftList = nums.slice(0 until n / 2)
    val rightList = nums.slice(n / 2 until n)
    val leftSubsets = generateSubsets(leftList)
    val rightSubsets = generateSubsets(rightList)

    val rightMap = mutableMapOf<Int, Int>()
    val leftMap = mutableMapOf<Int, Int>()
    for ((sum, _) in leftSubsets) {
        leftMap[sum] = leftMap.getOrDefault(sum, 0) + 1
    }
    for ((sum, _) in rightSubsets) {
        rightMap[sum] = rightMap.getOrDefault(sum, 0) + 1
    }

    // s1 - s2 = 2s1 - s = 2*l + 2*r - s = target
    //  ( target + s - 2 * l ) / 2

    var totalCases = 0
    for ((left, leftCount) in leftMap) {
        val up = target + totalSum - 2 * left
        if (up % 2 != 0) continue
        val right = up / 2
        val rightCount = rightMap[right] ?: continue
        // val leftCount = leftMap[left] ?: continue
        //  println("$leftCount $rightCount")
        totalCases += leftCount * rightCount
    }

    return totalCases
}

fun distributeCookies(cookies: IntArray, k: Int): Int {
    val n = cookies.size
    val children = IntArray(k)
    var minFair = Int.MAX_VALUE

    cookies.sortDescending()

    fun backtrack(position: Int, maxSoFar: Int) {
        if (maxSoFar >= minFair) return

        if (position == n) {
            minFair = maxSoFar
            return
        }

        for (i in 0 until k) {
            if (children[i] + cookies[position] >= minFair) continue

            children[i] += cookies[position]
            val newMax = maxOf(maxSoFar, children[i])

            backtrack(position + 1, newMax)

            children[i] -= cookies[position]

            if (children[i] == 0) break
        }
    }

    backtrack(0, 0)
    return minFair
}

fun minimumTimeRequired(jobs: IntArray, k: Int): Int {
    val n = jobs.size
    val workers = IntArray(k)
    var minFair = Int.MAX_VALUE

    jobs.sortDescending()

    fun backtrack(position: Int, maxSoFar: Int) {
        if (maxSoFar >= minFair) return

        if (position == n) {
            minFair = maxSoFar
            return
        }

        for (i in 0 until k) {
            if (workers[i] + jobs[position] >= minFair) continue

            workers[i] += jobs[position]
            val newMax = maxOf(maxSoFar, workers[i])

            backtrack(position + 1, newMax)

            workers[i] -= jobs[position]

            if (workers[i] == 0) break
        }
    }

    backtrack(0, 0)
    return minFair
}

fun canPartitionKSubsets(nums: IntArray, k: Int): Boolean {
    val n = nums.size
    val parts = IntArray(k)
    val totalSum = nums.sum()
    if (totalSum % k != 0) return false
    val partSum = totalSum / k

    nums.sortDescending()

    var found = false
    fun backtrack(position: Int) {
        if (found) return
        if (position == n) {
            if (parts.all { it == partSum }) {
                found = true
            }
            return
        }

        for (i in 0 until k) {
            if (parts[i] + nums[position] > partSum) continue

            parts[i] += nums[position]

            backtrack(position + 1)

            parts[i] -= nums[position]

            if (parts[i] == 0) break
        }
    }

    backtrack(0)
    return found
}

fun makesquare(matchsticks: IntArray): Boolean {
    val nums = matchsticks.map { it.toLong() }.sortedDescending()
    val n = nums.size
    val k = 4
    val parts = LongArray(k)
    val totalSum = nums.sum()
    if (totalSum % k != 0L) return false
    val partSum = totalSum / k

    var found = false
    fun backtrack(position: Int) {
        if (found) return
        if (position == n) {
            if (parts.all { it == partSum }) {
                found = true
            }
            return
        }

        for (i in 0 until k) {
            if (parts[i] + nums[position] > partSum) continue

            parts[i] += nums[position]

            backtrack(position + 1)

            parts[i] -= nums[position]

            if (parts[i] == 0L) break
        }
    }

    backtrack(0)
    return found
}

fun maxStrength(nums: IntArray): Long {
    val numbers = nums.map { it.toLong() }
    var result = numbers.max()
    val n = numbers.size
    for (mask in 0 until (1 shl n)) {
        var p = 1L
        for (i in 0 until n) {
            if (mask and (1 shl i) != 0) {
                p *= numbers[i]
            }
        }
        result = maxOf(result, p)
    }
    return result
}

fun beautifulSubsets(numbers: IntArray, k: Int): Int {
    var count = 0
    val n = numbers.size
    for (mask in 1 until (1 shl n)) {
        val set = mutableSetOf<Int>()
        var invalid = false
        for (i in 0 until n) {
            if (mask and (1 shl i) == 0) continue
            val num = numbers[i]
            if ((num + k) in set || (num - k) in set) {
                invalid = true
                break
            }
            set.add(num)
        }
        if (!invalid) {
            //  println(set)
            count++
        }
    }
    return count
}

fun constructDistancedSequence(n: Int): IntArray {
    val size = 2 * n - 1
    val maxValue = n + 1
    val seq = IntArray(size) { maxValue }
    val used = BooleanArray(n + 1)
    var maxSequence = IntArray(size) { 0 }

    fun compareArrays(a: IntArray, b: IntArray): Int {
        val minLen = minOf(a.size, b.size)
        for (i in 0 until minLen) {
            if (a[i] != b[i]) return a[i] - b[i] // >0 nghĩa là a > b
        }
        return a.size - b.size
    }

    fun backtrack() {
        val pos = seq.indexOfFirst { it == maxValue }

        if (compareArrays(maxSequence, seq) > 0) return

        if (pos < 0) {
            maxSequence = seq.copyOf()
            return
        }

        for (i in n downTo 1) {
            if (used[i]) continue
            if (seq[pos] != maxValue) continue
            if (i > 1 && pos + i < size && seq[pos + i] != maxValue) continue

            used[i] = true
            seq[pos] = i
            if (i > 1 && pos + i < size) {
                seq[pos + i] = i
            }

            backtrack()


            if (i > 1 && pos + i < size && seq[pos + i] == seq[pos]) {
                seq[pos + i] = maxValue
            }
            if (i > 1 && pos - i >= 0 && seq[pos - i] == seq[pos]) {
                seq[pos - i] = maxValue
            }
            seq[pos] = maxValue
            used[i] = false
        }
    }

    backtrack()
    return maxSequence
}


fun maxCompatibilitySum(students: Array<IntArray>, mentors: Array<IntArray>): Int {
    val m = students.size
    val n = students[0].size
    val scores = Array(m) { IntArray(m) }
    var maxCompatibility = 0
    for (i in 0 until m) {
        for (j in 0 until m) {
            var cnt = 0
            for (k in 0 until n) {
                if (students[i][k] == mentors[j][k]) {
                    cnt++
                }
            }
            scores[i][j] = cnt
            maxCompatibility = maxOf(maxCompatibility, cnt)
        }
    }

    var maxScore = 0
    val used = BooleanArray(m)

    fun backtrack(pos: Int, score: Int) {
        if (score + maxCompatibility * (m - pos) < maxScore) return
        if (pos == m) {
            if (score > maxScore) {
                maxScore = score
            }
            return
        }

        for (i in 0 until m) {
            if (used[i]) continue
            val nextScore = score + scores[pos][i]
            used[i] = true
            backtrack(pos + 1, nextScore)
            used[i] = false
        }
    }

    backtrack(0, 0)

    return maxScore
}

fun findDifferentBinaryString(nums: Array<String>): String {
    val set = nums.toSet()
    val n = nums[0].length
    val total = (1 shl n) - 1
    for (num in total downTo 0) {
        val binaryStr = num.toString(2).padStart(n, '0')
        if (binaryStr !in set) return binaryStr
    }
    return ""
}

fun maxScore(nums: IntArray): Int {
    fun gcd(a: Int, b: Int): Int {
        var x = abs(a)
        var y = abs(b)
        if (x == 0) return y
        if (y == 0) return x
        val shift = Integer.numberOfTrailingZeros(x or y)
        x = x shr Integer.numberOfTrailingZeros(x)
        while (y != 0) {
            y = y shr Integer.numberOfTrailingZeros(y)
            if (x > y) {
                val tmp = x
                x = y
                y = tmp
            }
            y -= x
        }
        return x shl shift
    }

    val size = nums.size
    val n = size / 2

    nums.sortDescending()
    val gcdList = Array(size) { IntArray(size) }
    var maxGcd = 0
    val maxRows = IntArray(size)

    for (i in 0 until size - 1) {
        for (j in i + 1 until size) {
            val a = nums[i]
            val b = nums[j]
            val g = gcd(a, b)
            gcdList[i][j] = g
            gcdList[j][i] = g
            maxGcd = maxOf(maxGcd, g)
            maxRows[i] = maxOf(maxRows[i], g)
        }
    }


    var maxScore = -1
    val used = BooleanArray(size)

    fun backtrack(pos: Int, score: Int) {
        var maxGCD = 0
        //   var remainMax = 0
        for (i in 0 until size) {
            if (used[i]) continue
            maxGCD = maxOf(maxGCD, maxRows[i])
//            for (j in i + 1 until size) {
//                if (used[j]) continue
//                remainMax = maxOf(remainMax, gcdList[i][j])
//            }
        }
        val maxRemain = maxGCD * (pos * (pos + 1)) / 2
        if (maxRemain + score <= maxScore) return

        if (pos == 0) {
            if (score > maxScore) {
                maxScore = score
            }
            return
        }

        for (i in 0 until size - 1) {
            if (used[i]) continue
            used[i] = true
            for (j in i + 1 until size) {
                if (used[j]) continue
                used[j] = true
                val newScore = score + pos * gcdList[i][j]
                backtrack(pos - 1, newScore)
                used[j] = false
            }
            used[i] = false
        }
    }
    backtrack(n, 0)
    return maxScore
}

fun permuteUnique(nums: IntArray): List<List<Int>> {
    val n = nums.size

    val result = mutableSetOf<List<Int>>()
    val used = BooleanArray(n)
    val list = mutableListOf<Int>()
    fun backtrack(pos: Int) {
        if (pos == n) {
            result.add(list)
            return
        }

        for (i in 0 until n) {
            if (used[i]) continue
            used[i] = true
            list.add(nums[i])
            backtrack(pos + 1)
            used[i] = false
            list.removeLast()
        }
    }
    backtrack(0)
    return result.toList()
}

fun restoreIpAddresses(s: String): List<String> {
    if (s.length !in 4..12) return emptyList()
    val n = s.length

    val result = mutableListOf<String>()
    for (i in 0 until 3) {
        val first = s.substring(0, i + 1)
        val firstNum = first.toInt()
        if (firstNum !in 0..255 || first != firstNum.toString()) break

        for (j in i + 1 until minOf(i + 4, n)) {
            val second = s.substring(i + 1, j + 1)
            val secondNum = second.toInt()
            if (secondNum !in 0..255 || second != secondNum.toString()) break

            for (k in j + 1 until minOf(j + 4, n)) {
                val third = s.substring(j + 1, k + 1)
                val thirdNum = third.toInt()
                if (thirdNum !in 0..255 || third != thirdNum.toString()) break

                if (s.length <= k + 1) break
                val forth = s.substring(k + 1, s.length)
                if (forth.length > 3) continue
                val forthNum = forth.toInt()
                if (forthNum !in 0..255 || forth != forthNum.toString()) continue

                val ip = "$first.$second.$third.$forth"
                result.add(ip)
            }
        }
    }

    return result
}

fun grayCode(n: Int): List<Int> {
    return List(1 shl n) {
        it xor (it shr 1)
    }
}

fun combine(n: Int, k: Int): List<List<Int>> {
    val used = BooleanArray(n + 1)
    val combination = IntArray(k)
    val result = mutableSetOf<List<Int>>()

    fun backtrack(pos: Int, minSoFar: Int) {
        if (pos == k) {
            result.add(combination.toList())
            return
        }

        for (i in (minSoFar + 1)..n) {
            if (used[i]) continue
            used[i] = true
            combination[pos] = i
            backtrack(pos + 1, i)
            used[i] = false
        }
    }

    backtrack(0, 0)

    return result.toList()
}

fun getMaximumGold(grid: Array<IntArray>): Int {
    val m = grid.size
    val n = grid[0].size


    var maxValue = 0
    val visited = Array(m) { BooleanArray(n) }
    fun backtrack(x: Int, y: Int, value: Int) {
        if (x !in 0 until m
            || y !in 0 until n
            || grid[x][y] == 0 || visited[x][y]
        ) {
            maxValue = maxOf(value, maxValue)
            return
        }

        visited[x][y] = true
        val newValue = value + grid[x][y]
        backtrack(x + 1, y, newValue)
        backtrack(x - 1, y, newValue)
        backtrack(x, y + 1, newValue)
        backtrack(x, y - 1, newValue)
        visited[x][y] = false
    }


    for (i in 0 until m) {
        visited[i] = BooleanArray(n)
        for (j in 0 until n) {
            backtrack(i, j, 0)
        }
    }
    return maxValue
}

fun combinationSum3(k: Int, n: Int): List<List<Int>> {
    val used = BooleanArray(10)
    val combination = IntArray(k)
    val result = mutableSetOf<List<Int>>()
    var currentSum = 0

    fun backtrack(pos: Int, maxSoFar: Int) {
        val r = k - pos
        if (currentSum + r > n) return
        if (currentSum + r * (maxSoFar - 1) < n) return
        if (pos == k && currentSum == n) {
            result.add(combination.toList())
            return
        }

        for (i in (maxSoFar - 1) downTo 1) {
            if (used[i]) continue
            used[i] = true
            combination[pos] = i
            currentSum += i
            backtrack(pos + 1, i)
            used[i] = false
            currentSum -= i
        }
    }

    backtrack(0, 10)

    return result.toList()
}

fun largestGoodInteger(num: String): String {
    val n = num.length
    if (n < 3) return ""

    var maxDigit = ""
    for (i in 0 until n - 2) {
        if (num[i] != num[i + 1] || num[i] != num[i + 2]) continue
        if (maxDigit.isEmpty() || (num[i] > maxDigit[0])) {
            maxDigit = "${num[i]}${num[i]}${num[i]}"
        }
    }
    return maxDigit
}

fun solveNQueens(n: Int): List<List<String>> {
    val result = mutableListOf<List<String>>()
    val board = Array(n) { CharArray(n) { '.' } }

    val cols = BooleanArray(n)
    val firstDiagonal = BooleanArray(2 * n) // row - col + n
    val secondDiagonal = BooleanArray(2 * n) // row + col

    fun backtrack(row: Int) {

        if (row == n) {
            result.add(board.map { String(it) })
            return
        }

        for (col in 0 until n) {
            val diagonal1 = row - col + n
            val diagonal2 = row + col
            if (cols[col] || firstDiagonal[diagonal1] || secondDiagonal[diagonal2]) continue
            board[row][col] = 'Q'
            cols[col] = true
            firstDiagonal[diagonal1] = true
            secondDiagonal[diagonal2] = true
            backtrack(row + 1)
            board[row][col] = '.'
            cols[col] = false
            firstDiagonal[diagonal1] = false
            secondDiagonal[diagonal2] = false
        }
    }

    backtrack(0)
    return result
}

fun minimumBeautifulSubstrings(s: String): Int {
    if (s[0] == '0') return -1
    val length = s.length
    val num = s.toInt(2)
    fun isPowerOfFive(n: Int): Boolean {
        return n > 0 && 15625 % n == 0
    }

    if (isPowerOfFive(num)) return 1

    fun extractBits(num: Int, i: Int, j: Int): Int {
        return (num shr i) and ((1 shl (j - i + 1)) - 1)
    }

    fun extractBitsFromLeft(num: Int, i: Int, j: Int): Int {
        val totalBits = 32 - Integer.numberOfLeadingZeros(num)
        val iFromRight = totalBits - 1 - i
        val jFromRight = totalBits - 1 - j
        val start = minOf(iFromRight, jFromRight)
        val end = maxOf(iFromRight, jFromRight)
        return (num shr start) and ((1 shl (end - start + 1)) - 1)
    }

    val candidates = (1 until length).filter { s[it] == '1' }
    val n = candidates.size

    val total = 1 shl n
    var minCount = length + 1

    for (mask in 1 until total) {
        var fromIndex = 0
        var invalid = false
        var cnt = 0
        for (i in 0 until n) {
            if (mask and (1 shl i) == 0) continue
            val toIndex = candidates[i]
            val sub = extractBitsFromLeft(num, fromIndex, toIndex - 1)

            if (!isPowerOfFive(sub)) {
                invalid = true
                break
            }
            cnt++
            if (cnt >= minCount) {
                invalid = true
                break
            }
            fromIndex = toIndex
        }
        if (invalid) continue

        val sub = extractBitsFromLeft(num, fromIndex, length - 1)

        if (!isPowerOfFive(sub)) continue
        cnt++
        minCount = minOf(minCount, cnt)
    }

    return if (minCount > length) -1 else minCount
}

fun letterCasePermutation(s: String): List<String> {
    val length = s.length
    val total = 1 shl length
    val result = mutableSetOf<String>()
    for (mask in 0 until total) {
        val str = StringBuilder()
        for (i in 0 until length) {
            if (mask and (1 shl i) == 0) {
                str.append(s[i].lowercase())
            } else {
                str.append(s[i].uppercase())
            }
        }
        result.add(str.toString())
    }
    return result.toList()
}

fun maximumEvenSplit(finalSum: Long): List<Long> {
    if (finalSum % 2 != 0L) return emptyList()

    val k = (sqrt(0.25 + finalSum) - 0.5).toInt()
    val lastNum = 2L * k + finalSum - k.toLong() * (1L + k)

    //  println("$finalSum ${k} ${finalSum - k * (k + 1)}")

    return List(k) {
        if (it < k - 1) 2L * it + 2L else lastNum
    }
}

fun minIncrementForUnique(nums: IntArray): Int {
    val maxValue = nums.max()
    val minValue = nums.min()
    val counts = IntArray(maxValue + 1)

    for (num in nums) {
        counts[num]++
    }

    var moves = 0
    val queue = ArrayDeque<Int>()
    var i = minValue
    while (i <= maxValue) {
        var count = counts[i]
        if (count == 0 && queue.isNotEmpty()) {
            moves += (i - queue.removeFirst())
        }
        while (count-- > 1) queue.add(i)
        i++
    }

    while (queue.isNotEmpty()) {
        moves += (i - queue.removeFirst())
        i++
    }
    return moves
}

fun maximumProduct(nums: IntArray, k: Int): Int {
    val mod = 1_000_000_007
    var minValue = Int.MAX_VALUE
    var maxValue = Int.MIN_VALUE
    for (num in nums) {
        minValue = minOf(minValue, num)
        maxValue = maxOf(maxValue, num)
    }

    val counts = mutableMapOf<Int, Int>()
    for (num in nums) {
        counts[num] = counts.getOrDefault(num, 0) + 1
    }

    if (counts.getOrDefault(0, 0) > k) return 0


    var moves = k
    var i = minValue

    while (moves > 0) {
        while (counts.getOrDefault(i, 0) > 0 && moves > 0) {
            counts[i] = counts.getOrDefault(i, 0) - 1
            counts[i + 1] = counts.getOrDefault(i + 1, 0) + 1
            moves--
        }
        i++
    }

    var product = 1L
    for (num in minValue..(maxValue + k)) {
        while (counts.getOrDefault(num, 0) > 0) {
            counts[num] = counts.getOrDefault(num, 0) - 1
            product = (product * num.toLong()) % mod
        }
    }

    return (product % mod).toInt()
}

fun generateParenthesis(n: Int): List<String> {
    val size = 2 * n
    val total = 1 shl size
    val result = mutableListOf<String>()
    for (mask in 1 until total) {
        if (Integer.bitCount(mask) != n) continue
        var balance = 0
        val sub = StringBuilder()
        for (i in 0 until size) {
            if (mask and (1 shl i) == 0) {
                sub.append('(')
                balance++
            } else {
                sub.append(')')
                balance--
            }
            if (balance < 0) break
        }
        if (balance != 0) continue
        result.add(sub.toString())
    }
    return result

}

fun letterCombinations(digits: String): List<String> {
    if (digits.isEmpty()) return emptyList()

    val characters = Array(10) { CharArray(4) { '.' } }
    characters[2] = charArrayOf('a', 'b', 'c', '.')
    characters[3] = charArrayOf('d', 'e', 'f', '.')
    characters[4] = charArrayOf('g', 'h', 'i', '.')
    characters[5] = charArrayOf('j', 'k', 'l', '.')
    characters[6] = charArrayOf('m', 'n', 'o', '.')
    characters[7] = charArrayOf('p', 'q', 'r', 's')
    characters[8] = charArrayOf('t', 'u', 'v', '.')
    characters[9] = charArrayOf('w', 'x', 'y', 'z')

    val length = digits.length
    val s = CharArray(length)
    val result = mutableListOf<String>()
    fun backtrack(pos: Int) {
        if (pos == length) {
            result.add(String(s))
            return
        }

        val digit = digits[pos] - '0'

        for (c in characters[digit]) {
            if (c == '.') continue
            s[pos] = c
            backtrack(pos + 1)
        }
    }
    backtrack(0)
    return result
}

fun numsSameConsecDiff(n: Int, k: Int): IntArray {
    val result = mutableSetOf<Int>()
    fun backtrack(pos: Int, num: Int) {
        if (pos == n) {
            result.add(num)
            return
        }

        val lastDigit = num % 10
        for (digit in intArrayOf(lastDigit + k, lastDigit - k)) {
            if (digit !in 0..9) continue
            backtrack(pos + 1, num * 10 + digit)
        }
    }
    for (i in 1..9) {
        backtrack(1, i)
    }
    return result.toIntArray()
}

fun uniquePathsIII(grid: Array<IntArray>): Int {
    val m = grid.size
    val n = grid[0].size
    val masks = IntArray(m)
    var startX = 0
    var startY = 0

    for (i in 0 until m) {
        for (j in 0 until n) {
            when (grid[i][j]) {
                -1 -> {
                    masks[i] = masks[i] or (1 shl j)
                }

                1 -> {
                    startX = i
                    startY = j
                }
            }
        }
    }

    //  println(masks.map { it.toString(2) })

    var cnt = 0
    fun dfs(x: Int, y: Int) {
        if (x !in 0 until m || y !in 0 until n || grid[x][y] == -1) return
        if (masks[x] and (1 shl y) != 0) return

        masks[x] = masks[x] or (1 shl y)

        if (grid[x][y] == 2) {
            //  println(masks.map { it.toString(2) })
            val target = (1 shl n) - 1
            if (masks.all { it == target }) {
                cnt++
            }
            masks[x] = masks[x] and (1 shl y).inv()
            return
        }

        dfs(x + 1, y)
        dfs(x - 1, y)
        dfs(x, y + 1)
        dfs(x, y - 1)
        // println(masks[x].toString(2))
        masks[x] = masks[x] and (1 shl y).inv()
        //  println(masks[x].toString(2))
    }

    dfs(startX, startY)
    return cnt
}

fun numTilePossibilities(tiles: String): Int {
    val freqMap = tiles.groupingBy { it }.eachCount().toList()
    val k = freqMap.size
    val freqs = freqMap.map { it.second }.toIntArray()
    val total = freqs.sum()

    val fact = intArrayOf(1, 1, 2, 6, 24, 120, 720, 5040)

    var totalCount = 0
    val chosen = IntArray(k)

    fun backtrack(pos: Int, remain: Int) {
        if (pos == k) {
            if (remain == 0) {
                val m = chosen.sum()
                if (m > 0) {
                    var denom = 1
                    for (c in chosen) denom *= fact[c]
                    totalCount += fact[m] / denom
                }
            }
            return
        }
        for (take in 0..minOf(freqs[pos], remain)) {
            chosen[pos] = take
            backtrack(pos + 1, remain - take)
        }
        chosen[pos] = 0
    }

    for (len in 1..total) {
        backtrack(0, len)
    }
    return totalCount
}

fun getProbability(balls: IntArray): Double {
    val k = balls.size
    // weights w_i = n_i - 1
    val w = IntArray(k) { balls[it] - 1 }

    // compute e_s (elementary symmetric sums) up to s=k
    val e = LongArray(k + 1)
    e[0] = 1L
    for (i in 0 until k) {
        val wi = w[i].toLong()
        for (s in i + 1 downTo 1) {
            e[s] += e[s - 1] * wi
        }
    }

    // Precompute binomial coefficients C(n,r) up to n=k (Pascal)
    val C = Array(k + 1) { LongArray(k + 1) }
    for (i in 0..k) {
        C[i][0] = 1L
        C[i][i] = 1L
        for (j in 1 until i) {
            C[i][j] = C[i - 1][j - 1] + C[i - 1][j]
        }
    }

    var total = 0L
    for (s in 0..k) {
        val rem = k - s
        if (rem % 2 != 0) continue
        val a = rem / 2
        // contribution = C(k - s, a) * e[s]
        total += C[rem][a] * e[s]
    }
    return total.toDouble() / (1L shl balls.sum())
}

fun closestCost(baseCosts: IntArray, toppingCosts: IntArray, target: Int): Int {
    var minDiff = Int.MAX_VALUE
    var minCost = Int.MAX_VALUE

    val doubleToppings = mutableListOf<Int>()
    for (cost in toppingCosts) {
        doubleToppings.add(cost * 2)
    }

    val maxCost = baseCosts.max() + doubleToppings.sum()
    if (maxCost <= target) return maxCost


    val queue = PriorityQueue<Pair<Int, Int>>(compareBy { abs(it.first - target) })
    for (base in baseCosts) {
        queue.add(base to 0)
//        for (i in toppingCosts.indices) {
//            val mask = 1 shl i
//            val cost = base + toppingCosts[i]
//            queue.add(cost to mask)
//        }
//
//        for (i in doubleToppings.indices) {
//            val mask = 1 shl i
//            val cost = base + doubleToppings[i]
//            queue.add(cost to mask)
//        }
    }

    while (queue.isNotEmpty()) {
        val (currentCost, currentMask) = queue.poll()

        if (currentCost == target) {
            return currentCost
        }
        val diff = abs(currentCost - target)
        if (diff < minDiff) {
            minDiff = diff
            minCost = currentCost
        } else if (diff == minDiff) {
            minCost = minOf(currentCost, minCost)
        }
        if (currentMask > 0) {

        }

        if (currentCost > target && currentCost - target > minDiff) {
            continue
        }

        for (i in toppingCosts.indices) {
            val bitwise = 1 shl i
            if (currentMask and bitwise != 0) continue

            val mask = currentMask or bitwise
            val cost = currentCost + toppingCosts[i]
            if (abs(cost - target) < abs(currentCost - target)) {
                queue.add(cost to mask)
            }
        }

        for (i in doubleToppings.indices) {
            val bitwise = 1 shl i
            if (currentMask and bitwise != 0) continue

            val mask = currentMask or bitwise
            val cost = currentCost + doubleToppings[i]
            if (abs(cost - target) <= abs(currentCost - target)) {
                queue.add(cost to mask)
            }
        }
    }

    return minCost
}


fun kthFactor(n: Int, k: Int): Int {
    val s = sqrt(n.toFloat()).toInt()
    val factors = mutableListOf<Int>()
    var cnt = 0
    for (i in 1..s) {
        if (n % i == 0) {
            cnt++
            if (cnt == k) {
                return i
            }
            factors.add(i)
        }
    }
    //  println(factors)
    //  val total = 2 * factors.size - if (s * s == n) 1 else 0
    //   if (k !in 1..total) return -1
//    cnt + (k - cnt)
//    cnt - (k - cnt)
//    cnt + (k - cnt)
//    cnt - (k - cnt - 1)
    val index = 2 * cnt - (if (s * s != n) k - 1 else k)
    return if (index > 0) (n / factors[index - 1]) else -1
}

fun reachNumber(target: Int): Int {
    val m = abs(target.toLong())

    var step = 0L
    var s = 0L
    while (s < m || (s - m) % 2 != 0L) {
        step++
        s = step * (step + 1L) / 2L
    }
    return step.toInt()
}

fun kthLargestValue(matrix: Array<IntArray>, k: Int): Int {
    val m = matrix.size
    val n = matrix[0].size
    val grid = Array(m) { IntArray(n) }
    val list = mutableListOf<Int>()
    val pq = PriorityQueue<Int>()
    for (i in 0 until m) {
        for (j in 0 until n) {
            val top = if (i > 0) grid[i - 1][j] else 0
            val left = if (j > 0) grid[i][j - 1] else 0
            val diagonal = if (i > 0 && j > 0) grid[i - 1][j - 1] else 0
            grid[i][j] = matrix[i][j] xor top xor left xor diagonal
            pq.add(grid[i][j])
            if (pq.size > k) pq.poll()
        }
    }
    //  println(grid.joinToString("\n") { it.toList().toString() })
    return pq.peek()
}

fun kthLargest(arr: IntArray, k: Int): Int {
    var left = 0
    var right = arr.lastIndex
    val target = arr.size - k

    fun partition(l: Int, r: Int): Int {
        val pivot = arr[r]
        var i = l
        for (j in l until r) {
            if (arr[j] <= pivot) {
                arr[i] = arr[j].also { arr[j] = arr[i] }
                i++
            }
        }
        arr[i] = arr[r].also { arr[r] = arr[i] }
        return i
    }

    while (true) {
        val p = partition(left, right)
        when {
            p == target -> return arr[p]
            p < target -> left = p + 1
            else -> right = p - 1
        }
    }
}

fun kthLargestNumber2(arr: Array<String>, k: Int): String {

    fun compare(first: String, second: String): Int {
        val a = first.trimStart('0')
        val b = second.trimStart('0')

        if (a.length != b.length) {
            return a.length.compareTo(b.length)
        }
        for (i in a.indices) {
            if (a[i] != b[i]) {
                return a[i].compareTo(b[i])
            }
        }
        return 0
    }

    val compare = mutableMapOf<Pair<String, String>, Int>()

    for (i in 0 until arr.size) {
        for (j in i + 1 until arr.size) {
            val a = arr[i]
            val b = arr[j]
            val result = compare(a, b)
            compare[a to b] = result
            compare[b to a] = -result
        }
    }

    var left = 0
    var right = arr.lastIndex
    val target = arr.size - k

    fun partition(l: Int, r: Int): Int {
        val pivot = arr[r]
        var i = l
        for (j in l until r) {
            if ((compare[arr[j] to pivot] ?: 0) <= 0) {
                arr[i] = arr[j].also { arr[j] = arr[i] }
                i++
            }
        }
        arr[i] = arr[r].also { arr[r] = arr[i] }
        return i
    }

    while (true) {
        val p = partition(left, right)
        when {
            p == target -> return arr[p]
            p < target -> left = p + 1
            else -> right = p - 1
        }
    }

}

fun kthLargestNumber(arr: Array<String>, k: Int): String {
    val pq = PriorityQueue<String>(compareBy<String> { it.length }.thenBy { it })
    for (num in arr) {
        pq.add(num)
        if (pq.size > k) pq.poll()
    }
    return pq.peek()
}

fun smallestTrimmedNumbers(nums: Array<String>, queries: Array<IntArray>): IntArray {

    val answers = IntArray(queries.size)
    for (i in queries.indices) {
        val (k, n) = queries[i]
        val pq = PriorityQueue<Pair<String, Int>>(
            compareByDescending<Pair<String, Int>> { it.first.length }
                .thenByDescending { it.first }
                .thenByDescending { it.second }
        )
        for (i in nums.indices) {
            pq.add(nums[i].takeLast(n) to i)
            if (pq.size > k) pq.poll()
        }
        //  println(pq)
        answers[i] = pq.peek().second
    }
    return answers
}

fun kthSmallestPrimeFraction(arr: IntArray, k: Int): IntArray {

    // Fraction IntArray { numerator, denominator, numeratorIndex, denominator)
    val n = arr.size
    val pq = PriorityQueue<IntArray>(compareBy { (num, den, _, _) ->
        num.toDouble() / den.toDouble()
    })

    for (i in 1 until n) {
        val fraction = intArrayOf(arr[0], arr[i], 0, i)
        pq.add(fraction)
    }
    repeat(k - 1) {
        val (_, den, numIndex, denoIndex) = pq.poll()

        val nextIndex = numIndex + 1
        if (nextIndex < denoIndex) {
            val fraction = intArrayOf(arr[nextIndex], den, nextIndex, denoIndex)
            pq.add(fraction)
        }
    }

    val (num, den) = pq.peek()
    return intArrayOf(num, den)
}

fun kthSmallest(matrix: Array<IntArray>, k: Int): Int {
    // Element IntArray {  num, row, col)
    val n = matrix.size
    val pq = PriorityQueue<IntArray>(compareBy { (num, _, _) ->
        num
    })

    val used = Array(n) { BooleanArray(n) }

    for (i in 0 until n) {
        val element = intArrayOf(matrix[0][i], 0, i)
        used[0][i] = true
        pq.add(element)
    }
    repeat(k - 1) { count ->
        val (num, row, col) = pq.poll()
        //    println("${count + 1}: $num " + pq.map { it.first() })
        if (row + 1 < n && !used[row + 1][col]) {
            val next = intArrayOf(matrix[row + 1][col], row + 1, col)
            used[row + 1][col] = true
            pq.add(next)
        }

        if (col + 1 < n && !used[row][col + 1]) {
            val next = intArrayOf(matrix[row][col + 1], row, col + 1)
            used[row][col + 1] = true
            pq.add(next)
        }

//        when {
//            row + 1 < n && col + 1 < n -> {
//                val a = matrix[row + 1][col]
//                val b = matrix[row][col + 1]
//                val next = if (a > b) intArrayOf(a, row + 1, col) else intArrayOf(b, row, col + 1)
//                pq.add(next)
//            }
//
//            row + 1 < n -> {
//                val next = intArrayOf(matrix[row + 1][col], row + 1, col)
//                pq.add(next)
//            }
//
//            col + 1 < n -> {
//                val next = intArrayOf(matrix[row][col + 1], row, col + 1)
//                pq.add(next)
//            }
//        }

    }

    return pq.peek().first()
}

fun findKthNumber(m: Int, n: Int, k: Int): Int {
    fun countLessThan(x: Int): Int {
        println("Count $x")
        val c = floor(x.toFloat() / n)
        var cnt = 0
        for (i in 1..m) {
            var l = 1
            var r = n
            var result = -1
            while (l <= r) {
                val j = (l + r) / 2
                if (i * j <= x) {
                    l = j + 1
                    result = j
                } else r = j - 1
            }
            println()
            cnt += result
        }
        println("$x $cnt")
        return cnt
    }

    var low = 1
    var high = m * n
    var result = -1
    while (low <= high) {
        val mid = low + (high - low) / 2
        val count = countLessThan(mid)
        if (count >= k) {
            if (count == k) result = mid
            high = mid - 1
        } else {
            low = mid + 1
        }
    }

    return result
}

fun judgePoint24(cards: IntArray): Boolean {
    val digits = cards.map { it.digitToChar() }
    val operators = setOf('+', '-', '*', '/')
    fun precedence(op: Char): Int = when (op) {
        '+', '-' -> 1
        '*', '/' -> 2
        else -> 0
    }

    fun applyOp(a: Double, b: Double, op: Char): Double {
        return when (op) {
            '+' -> a + b
            '-' -> a - b
            '*' -> a * b
            '/' -> if (b == 0.0) Double.NaN else a / b
            else -> Double.NaN
        }
    }

    fun evalExpr(expr: CharArray): Double {
        val values = ArrayDeque<Double>()
        val ops = ArrayDeque<Char>()

        var i = 0
        while (i < expr.size) {
            val ch = expr[i]

            when {
                ch == '(' -> ops.addLast(ch)

                ch.isDigit() -> {
                    var num = 0.0
                    while (i < expr.size && expr[i].isDigit()) {
                        num = num * 10 + (expr[i] - '0')
                        i++
                    }
                    i--
                    values.addLast(num)
                }

                ch == ')' -> {
                    while (ops.isNotEmpty() && ops.last() != '(') {
                        if (values.size < 2) return Double.NaN
                        val b = values.removeLast()
                        val a = values.removeLast()
                        val op = ops.removeLast()
                        values.addLast(applyOp(a, b, op))
                    }
                    if (ops.isNotEmpty() && ops.last() == '(') ops.removeLast()
                }

                ch in operators -> {
                    while (ops.isNotEmpty() && precedence(ops.last()) >= precedence(ch)) {
                        if (values.size < 2) return Double.NaN
                        val b = values.removeLast()
                        val a = values.removeLast()
                        val op = ops.removeLast()
                        values.addLast(applyOp(a, b, op))
                    }
                    ops.addLast(ch)
                }
            }
            i++
        }

        while (ops.isNotEmpty()) {
            if (values.size < 2) return Double.NaN
            val b = values.removeLast()
            val a = values.removeLast()
            val op = ops.removeLast()
            values.addLast(applyOp(a, b, op))
        }

        return if (values.size == 1) values.last() else Double.NaN
    }


    val templates = listOf(
        charArrayOf('a', '?', 'b', '?', 'c', '?', 'd'),
        charArrayOf('(', 'a', '?', 'b', ')', '?', 'c', '?', 'd'),
        charArrayOf('a', '?', '(', 'b', '?', 'c', ')', '?', 'd'),
        charArrayOf('a', '?', 'b', '?', '(', 'c', '?', 'd', ')'),
        charArrayOf('(', 'a', '?', 'b', '?', 'c', ')', '?', 'd'),
        charArrayOf('a', '?', '(', 'b', '?', 'c', '?', 'd', ')'),
        charArrayOf('(', 'a', '?', 'b', ')', '?', '(', 'c', '?', 'd', ')'),
        charArrayOf('(', '(', 'a', '?', 'b', ')', '?', 'c', ')', '?', 'd'),
        charArrayOf('(', 'a', '?', '(', 'b', '?', 'c', ')', ')', '?', 'd'),
        charArrayOf('(', 'a', '?', 'b', ')', '?', 'c', '?', 'd'),
        charArrayOf('a', '?', '(', '(', 'b', '?', 'c', ')', '?', 'd', ')'),
        charArrayOf('a', '?', '(', 'b', '?', '(', 'c', '?', 'd', ')', ')'),
        charArrayOf('(', 'a', '?', 'b', '?', 'c', '?', 'd', ')'),
        charArrayOf('(', '(', 'a', '?', 'b', ')', '?', 'c', '?', 'd', ')'),
        charArrayOf('(', 'a', '?', '(', 'b', '?', 'c', ')', '?', 'd', ')'),
        charArrayOf('(', 'a', '?', 'b', '?', '(', 'c', '?', 'd', ')', ')'),
        charArrayOf('(', '(', 'a', '?', 'b', '?', 'c', ')', '?', 'd', ')'),
        charArrayOf('(', 'a', '?', '(', 'b', '?', 'c', '?', 'd', ')', ')'),
        charArrayOf('(', '(', 'a', '?', 'b', ')', '?', '(', 'c', '?', 'd', ')', ')'),
        charArrayOf('(', '(', '(', 'a', '?', 'b', ')', '?', 'c', ')', '?', 'd', ')'),
        charArrayOf('(', '(', 'a', '?', '(', 'b', '?', 'c', ')', ')', '?', 'd', ')'),
        charArrayOf('(', 'a', '?', '(', '(', 'b', '?', 'c', ')', '?', 'd', ')', ')'),
        charArrayOf('(', 'a', '?', '(', 'b', '?', '(', 'c', '?', 'd', ')', ')', ')')
    )

    // println(digits)
    var found = false
    fun backtrack(template: CharArray, pos: Int, expression: CharArray, usedDigit: Int) {
        if (found) return
        if (pos == template.size) {
            val value = evalExpr(expression)
            if (abs(value - 24.0) < 1e-6) {
                found = true
                return
            }
            //  println(String(expression) + " = $value")
            return
        }

        val token = template[pos]
        if (token == ')' || token == '(') {
            expression[pos] = token
            backtrack(template, pos + 1, expression, usedDigit)
            return
        }
        if (token in 'a'..'d') {
            for (i in 0 until digits.size) {
                val isUsed = usedDigit and (1 shl i) != 0
                if (isUsed) continue
                val nextUsed = usedDigit or (1 shl i)
                expression[pos] = digits[i]
                backtrack(template, pos + 1, expression, nextUsed)
            }
            return
        }

        if (token == '?') {
            for (op in operators) {
                expression[pos] = op
                backtrack(template, pos + 1, expression, usedDigit)
            }
        }
    }

    for (template in templates) {
        backtrack(template, 0, template.clone(), 0)
        if (found) return true
    }
    return false
}

fun isValid(s: String): Boolean {
    if (s.length % 2 != 0) return false
    val stack = Stack<Char>()
    val set = setOf('{' to '}', '[' to ']', '(' to ')')
    for (p in s) {
        if (stack.isNotEmpty()) {
            val top = stack.peek()
            if ((top to p) in set) {
                stack.pop()
                continue
            }
        }
        stack.add(p)
    }
    return stack.isEmpty()
}

fun removeInvalidParentheses(s: String): List<String> {
    val n = s.length
    var answers = mutableSetOf<String>()
    val result = mutableListOf<Char>()
    var minStep = n

    fun backtrack(pos: Int, balance: Int, removedCount: Int) {
        if (pos == n) {
            if (balance != 0) return
            if (removedCount < minStep) {
                answers = mutableSetOf(result.joinToString(""))
                minStep = removedCount
            } else if (removedCount == minStep) {
                answers.add(result.joinToString(""))
            }
            return
        }
        val ch = s[pos]
        if (ch != ')' && ch != '(') {
            //           val preCh = result.lastOrNull()
//            if (preCh != null && preCh != ')' && preCh != '(') {
//                return
//            }
            result.add(ch)
            backtrack(pos + 1, balance, removedCount)
            result.removeLast()
            return
        }

        val newBalance = balance + if (ch == '(') 1 else -1
        if (newBalance >= 0) {
            result.add(ch)
            backtrack(pos + 1, newBalance, removedCount)
            result.removeLast()
        }
        backtrack(pos + 1, balance, removedCount + 1)
    }

    backtrack(0, 0, 0)
    return answers.toList()
}

fun closestRoom(rooms: Array<IntArray>, queries: Array<IntArray>): IntArray {
    val n = rooms.size
    val tree = TreeMap<Int, MutableList<Int>>()

    fun findClosestIdIndex(ids: List<Int>, target: Int): Int {
        if (ids.isEmpty()) return -1

        if (ids.size == 1) return 0

        var left = 0
        var right = ids.size - 1

        if (target <= ids[0]) return 0
        if (target >= ids[right]) return right

        while (left <= right) {
            val mid = left + (right - left) / 2

            if (ids[mid] == target) return mid

            if (mid < ids.size - 1 && ids[mid] < target && target < ids[mid + 1]) {
                return if (target - ids[mid] <= ids[mid + 1] - target) mid else mid + 1
            }

            if (ids[mid] < target) {
                left = mid + 1
            } else {
                right = mid - 1
            }
        }

        return if (left > 0) {
            if (target - ids[left - 1] <= ids[left] - target) left - 1 else left
        } else {
            left
        }
    }

    for ((id, size) in rooms) {
        val entry = tree[size]
        if (entry == null) {
            tree[size] = mutableListOf(id)
        } else {
            tree[size]?.add(id)
        }
    }

    tree.forEach { it.value.sort() }
    val answers = IntArray(queries.size) { -1 }
    for (i in queries.indices) {
        val (preferId, minSize) = queries[i]
        val lowEntry = tree.ceilingEntry((minSize)) ?: continue
        val entries = tree.tailMap(lowEntry.key)

        var closestId = -1
        var closestDistance = Int.MAX_VALUE
        for ((_, ids) in entries) {
            val idx = ids.binarySearch(preferId).let {
                if (it < 0) -it - 1 else it
            }
            val cId = listOf(i - 1, i, i + 1).filter {
                it in 0 until ids.size
            }.minBy { abs(ids[it] - preferId) }
            val id = ids[cId]
            val distance = abs(id - preferId)
            if (distance < closestDistance) {
                closestDistance = distance
                closestId = id
            } else if (distance == closestDistance) {
                closestId = minOf(closestId, id)
            }
        }
        answers[i] = closestId
    }
    return answers
}

class DataStream(private val value: Int, private val k: Int) {
    private val queue = ArrayDeque<Int>()

    fun consec(num: Int): Boolean {
        if (k == 1) {
            if (num == value) {
                queue.clear()
                queue.add(num)
                return true
            }
            return false
        }

        if (num != value) {
            queue.clear()
            return false
        }

        queue.add(num)
        if (queue.size == k) {
            queue.removeFirst()
            return true
        }
        return false
    }
}

fun longestSubstring(s: String, k: Int): Int {
    val n = s.length
    val freq = Array(n + 1) { IntArray(26) }

    for (i in 0 until n) {
        for (c in 0 until 26) {
            freq[i + 1][c] = freq[i][c]
        }
        freq[i + 1][s[i] - 'a']++
    }

    fun getMinFreq(l: Int, r: Int): Pair<Char, Int> {
        var minChar = '?'
        var minFreq = Int.MAX_VALUE
        for (c in 0 until 26) {
            val freq = freq[r + 1][c] - freq[l][c]
            if (freq > 0 && freq < minFreq) {
                minFreq = freq
                minChar = ('a' + c)
            }
        }
        return minChar to minFreq
    }

    return 1
}

fun createSmallerLeft(nums: IntArray): IntArray {
    val n = nums.size
    val stack = Stack<Int>()
    stack.clear()
    val smallerLeft = IntArray(n) { -1 }
    for (i in (n - 1) downTo 0) {
        while (stack.isNotEmpty() && nums[i] < nums[stack.peek()]) {
            val end = stack.pop()
            smallerLeft[end] = i
        }
        stack.push(i)
    }
    return smallerLeft
}

fun createGreaterRight(nums: IntArray): IntArray {
    val n = nums.size
    val stack = Stack<Int>()
    val greaterRight = IntArray(n) { n }
    for (i in 0 until n) {
        while (stack.isNotEmpty() && nums[i] > nums[stack.peek()]) {
            val start = stack.pop()
            greaterRight[start] = i
        }
        stack.push(i)
    }
    return greaterRight
}

fun minCost(startPos: IntArray, homePos: IntArray, rowCosts: IntArray, colCosts: IntArray): Int {
    val m = rowCosts.size
    val n = colCosts.size


    val (startX, startY) = startPos
    val (homeX, homeY) = homePos

    val prefixRows = LongArray(m)
    prefixRows[0] = rowCosts[0].toLong()
    for (i in 1 until m) {
        prefixRows[i] = prefixRows[i - 1] + rowCosts[i].toLong()
    }

    val prefixCols = LongArray(n)
    prefixCols[0] = colCosts[0].toLong()
    for (j in 1 until n) {
        prefixCols[j] = prefixCols[j - 1] + colCosts[j].toLong()
    }


    val minX = if (homeX >= startX) startX else homeX - 1
    val maxX = if (homeX >= startX) homeX else startX - 1
    val minY = if (homeY >= startY) startY else homeY - 1
    val maxY = if (homeY >= startY) homeY else startY - 1

    val colCost = prefixCols[maxY] - (if (minY >= 0) prefixCols[minY] else 0L)
    val rowCost = prefixRows[maxX] - (if (minX >= 0) prefixRows[minX] else 0L)

    return (colCost + rowCost).toInt()
}

fun maxBalancedSubsequenceSum(nums: IntArray): Long {
    val n = nums.size
    var minValue = Long.MAX_VALUE
    var maxValue = Long.MIN_VALUE
    val balances = List(n) { index ->
        val balance = nums[index].toLong() - index.toLong()
        minValue = minOf(minValue, balance)
        maxValue = maxOf(maxValue, balance)
        balance
    }
    val maxSegmentTree = MaxDynamicSegmentTreeLong(minValue, maxValue)
    var result = Long.MIN_VALUE
    for (i in 0 until n) {
        val balance = balances[i]
        val d = maxSegmentTree.query(minValue, balance) + nums[i].toLong()
        result = maxOf(result, d)
        maxSegmentTree.update(balance, d)
    }
    return result
}

fun main() {
    // 1 2 4 8 16
    // 1 3 5 15
    val matrix = arrayOf(
        intArrayOf(1, 3, 5),
        intArrayOf(6, 7, 12),
        intArrayOf(11, 14, 14)
    )
    val k = 6
    println(
        //   findKthNumber(9895, 28405, 100787757)
        maxBalancedSubsequenceSum(intArrayOf(3,3,5,6))
    )
}