package org.example

import java.util.*
import kotlin.math.max

fun dailyTemperatures(temperatures: IntArray): IntArray {
    val n = temperatures.size
    val stack = Stack<Int>()
    val result = IntArray(n)

    for (i in 0 until n) {
        while (stack.isNotEmpty() && temperatures[i] > temperatures[stack.peek()]) {
            val start = stack.pop()
            result[start] = i - start
        }
        stack.push(i)
    }
    return result
}

fun trap(height: IntArray): Int {
    val n = height.size
    if (n < 3) return 0
    val stack = Stack<Int>()
    var total = 0

    val prefixSum = IntArray(n) { -1 }
    prefixSum[0] = height[0]
    for (i in 1 until n) {
        prefixSum[i] = prefixSum[i - 1] + height[i]
    }

    val greaterRight = IntArray(n) { n }
    for (i in 0 until n) {
        while (stack.isNotEmpty() && height[i] >= height[stack.peek()]) {
            val start = stack.pop()
            greaterRight[start] = i
        }
        stack.push(i)
    }

    stack.clear()
    val greaterLeft = IntArray(n)
    for (i in (n - 1) downTo 0) {
        while (stack.isNotEmpty() && height[i] >= height[stack.peek()]) {
            val end = stack.pop()
            greaterLeft[end] = i
        }
        stack.push(i)
    }

    var l = 0
    var r = n - 1

    while (true) {
        val left = greaterRight[l]
        val right = greaterLeft[r]
        if (left > right || left >= n || right < 0) break

        if (left > l + 1) {
            val leftSum = prefixSum[left - 1] - prefixSum[l]
            val volumeLeft = (left - l - 1) * height[l]
            val waterLeft = (volumeLeft - leftSum).coerceAtLeast(0)
            //   println("$l -> $left: $waterLeft")
            total += waterLeft
        }
        l = left

        if (r > right + 1) {
            val rightSum = prefixSum[r - 1] - prefixSum[right]
            val volumeRight = (r - 1 - right) * height[r]
            val waterRight = (volumeRight - rightSum).coerceAtLeast(0)
            // println("$r -> $right: $waterRight")
            total += waterRight
        }
        r = right
    }

    var leftMax = height[l]
    var rightMax = height[r]
    while (l < r) {
        if (leftMax <= rightMax) {
            l++
            if (height[l] < leftMax) {
                val volume = leftMax - height[l]
                total += volume
            } else {
                leftMax = height[l]
            }
        } else {
            r--
            if (height[r] < rightMax) {
                total += (rightMax - height[r])
            } else {
                rightMax = height[r]
            }
        }
    }
    return total
}

fun maximumTripletValue(nums: IntArray): Long {
    val n = nums.size
    val prefix = IntArray(n)
    val suffix = IntArray(n)

    prefix[0] = nums[0]
    for (i in 1 until n) {
        prefix[i] = maxOf(nums[i], prefix[i - 1])
    }
    suffix[n - 1] = nums.last()
    for (i in (n - 2) downTo 0) {
        suffix[i] = maxOf(nums[i], suffix[i + 1])
    }

    var maxValue = 0L

    for (i in 1 until (n - 1)) {
        val value = (prefix[i - 1] - nums[i]).toLong() * suffix[i + 1].toLong()
        maxValue = maxOf(maxValue, value)
    }
    return maxValue
}

fun minimumSum(nums: IntArray): Int {
    val n = nums.size
    val prefix = IntArray(n)
    val suffix = IntArray(n)

    prefix[0] = nums[0]
    for (i in 1 until n) {
        prefix[i] = minOf(nums[i], prefix[i - 1])
    }
    suffix[n - 1] = nums.last()
    for (i in (n - 2) downTo 0) {
        suffix[i] = minOf(nums[i], suffix[i + 1])
    }

    var minValue = Int.MAX_VALUE
    var found = false
    for (i in 1 until (n - 1)) {
        val left = prefix[i - 1]
        val right = suffix[i + 1]
        val num = nums[i]
        if (left < num && num > right) {
            found = true
            minValue = minOf(minValue, left + num + right)
        }
    }
    return if (found) minValue else -1
}

fun nextGreaterElements(nums: IntArray): IntArray {
    val n = nums.size
    val numbers = nums + nums
    val stack = Stack<Int>()

    val greaterRight = IntArray(n) { n }
    for (i in 0 until 2 * n) {
        while (stack.isNotEmpty() && numbers[i] > numbers[stack.peek()]) {
            val start = stack.pop()
            greaterRight[start] = if (i >= start + n) -1 else numbers[i]
        }
        if (i < n) stack.push(i)
    }
    while (stack.isNotEmpty()) {
        greaterRight[stack.pop() % n] = -1
    }
    return greaterRight
}

fun minMaxSubarraySum(nums: IntArray, k: Int): Long {
//    fun countSubarraysContainingI(start: Int, end: Int, k: Int, i: Int): Long {
//        return (1..minOf(k, end - start + 1)).sumOf { len ->
//            val lmin = maxOf(start, i - len + 1)
//            val lmax = minOf(i, end - len + 1)
//            maxOf(0, lmax - lmin + 1).toLong()
//        }
//    }

    fun count(a: Int, b: Int, k: Int): Long {
        val (x, y) = if (a < b) a to b else b to a
        return when {
            k <= x + 1 -> (k.toLong() + 1) * k / 2
            k <= y -> (x + 2L) * (x + 1) / 2 + (k - x - 1) * (x + 1).toLong()
            k <= x + y + 1 -> {
                val t1 = (x + 2L) * (x + 1) / 2
                val t2 = (y - x - 1) * (x + 1L)
                val len = x + y + 1 - (k - 1)
                val t3 = (x + 1L + len) * (k - y) / 2
                t1 + t2 + t3
            }

            else -> (x + 2L) * (x + 1) + (y - x - 1) * (x + 1L)
        }
    }

    fun countSubarraysContainingI(start: Int, end: Int, k: Int, i: Int): Long {
        val a = i - start
        val b = end - i
        return count(a, b, k)
    }

    val n = nums.size
    val stack = Stack<Int>()

    val greaterRight = IntArray(n) { n }
    for (i in 0 until n) {
        while (stack.isNotEmpty() && nums[i] >= nums[stack.peek()]) {
            val start = stack.pop()
            greaterRight[start] = i
        }
        stack.push(i)
    }

    stack.clear()
    val smallerRight = IntArray(n) { n }
    for (i in 0 until n) {
        while (stack.isNotEmpty() && nums[i] < nums[stack.peek()]) {
            val start = stack.pop()
            smallerRight[start] = i
        }
        stack.push(i)
    }

    stack.clear()
    val greaterLeft = IntArray(n) { -1 }
    for (i in (n - 1) downTo 0) {
        while (stack.isNotEmpty() && nums[i] > nums[stack.peek()]) {
            val end = stack.pop()
            greaterLeft[end] = i
        }
        stack.push(i)
    }

    stack.clear()
    val smallerLeft = IntArray(n) { -1 }
    for (i in (n - 1) downTo 0) {
        while (stack.isNotEmpty() && nums[i] <= nums[stack.peek()]) {
            val end = stack.pop()
            smallerLeft[end] = i
        }
        stack.push(i)
    }

    var totalSum = 0L
    for (i in 0 until n) {
        val minStart = greaterLeft[i] + 1
        val minEnd = greaterRight[i] - 1
        if (minStart <= i && i <= minEnd) {
            val minSubCount = countSubarraysContainingI(minStart, minEnd, k, i)
            totalSum += (minSubCount * nums[i])
        }

        val maxStart = smallerLeft[i] + 1
        val maxEnd = smallerRight[i] - 1
        if (maxStart <= i && i <= maxEnd) {
            val maxSubCount = countSubarraysContainingI(maxStart, maxEnd, k, i)
            totalSum += (maxSubCount * nums[i])
        }
    }
    return totalSum
}

fun hIndex(citations: IntArray): Int {
    val n = citations.size
    var left = 0
    var right = n - 1
    var result = 0
    while (left <= right) {
        val mid = (left + right) / 2
        val level = n - mid
        if (citations[mid] >= level) {
            result = level
            right = mid - 1
        } else {
            left = mid + 1
        }
    }
    return result
}

fun longestMountain(arr: IntArray): Int {
    val n = arr.size
    if (n < 2) return 0
    val left = IntArray(n)
    val right = IntArray(n)

    left[0] = 0
    for (i in 1 until n) {
        if (arr[i] > arr[i - 1]) {
            left[i] = left[i - 1] + 1
        } else {
            left[i] = 0
        }
    }

    right[n - 1] = 0
    for (i in (n - 2) downTo 0) {
        if (arr[i] > arr[i + 1]) {
            right[i] = right[i + 1] + 1
        } else {
            right[i] = 0
        }
    }

    // println(arr.toList())
    //  println(left.toList())
    //  println(right.toList())
    var maxLength = 0

    for (i in 1 until (n - 1)) {
        val l = left[i]
        val r = right[i]
        if (l > 0 && r > 0) {
            maxLength = maxOf(maxLength, r + l + 1)
        }
    }
    return maxLength
}

fun maximumAmount(coins: Array<IntArray>): Int {
    val m = coins.size
    val n = coins[0].size
    val x = 2

    val dp = Array(x + 1) { Array(m + 1) { IntArray(n + 1) { -500 * 2000 } } }
    for (k in 0..x) {
        dp[k][0][1] = 0
        dp[k][1][0] = 0
    }

    for (i in 1..m) {
        for (j in 1..n) {
            for (k in 0..x) {
                val coin = coins[i - 1][j - 1]
                dp[k][i][j] = maxOf(dp[k][i - 1][j], dp[k][i][j - 1]) + coin

                if (k > 0 && coin < 0) {
                    dp[k][i][j] = maxOf(
                        dp[k][i][j],
                        dp[k - 1][i - 1][j],
                        dp[k - 1][i][j - 1],
                    )
                }
            }
        }
    }
    //  println(dp[0].joinToString("\n") { it.toList().toString() })
    //  println()
    //  println(dp[1].joinToString("\n") { it.toList().toString() })
    //  println()
    //  println(dp[2].joinToString("\n") { it.toList().toString() })
    return maxOf(dp[1][m][n], dp[2][m][n], dp[0][m][n])
}

fun goodDaysToRobBank(security: IntArray, time: Int): List<Int> {
    val n = security.size

    if (n < 2 * time + 1) return emptyList()
    if (n == 1) return listOf(0)
    if (n == 2) return listOf(0, 1)

    val left = IntArray(n)
    val right = IntArray(n)

    left[0] = 0
    for (i in 1 until n) {
        if (security[i] <= security[i - 1]) {
            left[i] = left[i - 1] + 1
        } else {
            left[i] = 0
        }
    }

    right[n - 1] = 0
    for (i in (n - 2) downTo 0) {
        if (security[i] <= security[i + 1]) {
            right[i] = right[i + 1] + 1
        } else {
            right[i] = 0
        }
    }

    // println(arr.toList())
    //  println(left.toList())
    //  println(right.toList())
    val goodDays = mutableListOf<Int>()

    for (i in 0 until n) {
        val l = left[i]
        val r = right[i]
        if (l >= time && r >= time) {
            goodDays.add(i)
        }
    }
    return goodDays
}


fun finalPrices(prices: IntArray): IntArray {
    val n = prices.size
    val stack = Stack<Int>()
    val smallerRight = IntArray(n) { n }
    for (i in 0 until n) {
        while (stack.isNotEmpty() && prices[i] <= prices[stack.peek()]) {
            val start = stack.pop()
            smallerRight[start] = i
        }
        stack.push(i)
    }

    val answers = IntArray(n)

    var discount = 0
    for (i in 0 until n) {
        val minIndex = smallerRight[i]
        discount = if (minIndex < n) prices[minIndex] else 0
        answers[i] = (prices[i] - discount).coerceAtLeast(0)
    }
    return answers
}

fun largestRectangleArea(heights: IntArray): Int {
    val n = heights.size
    //  println(heights.toList())
    val stack = Stack<Int>()

    val left = IntArray(n)

    for (i in (n - 1) downTo 0) {
        while (stack.isNotEmpty() && heights[i] < heights[stack.peek()]) {
            left[stack.pop()] = i
        }
        stack.push(i)
    }

    while (stack.isNotEmpty()) {
        left[stack.pop()] = -1
    }

    // println(left.toList())


    val right = IntArray(n)
    for (i in 0 until n) {
        while (stack.isNotEmpty() && heights[i] < heights[stack.peek()]) {
            right[stack.pop()] = i
        }
        stack.push(i)
    }

    while (stack.isNotEmpty()) {
        right[stack.pop()] = n
    }
    // println(right.toList())

    var maxArea = 0

    for(i in 0 until n) {
        if(heights[i] > 0) {
            val area = heights[i] * (right[i] - left[i] - 1)
            maxArea = max(maxArea, area)
        }
    }

    return maxArea


}

fun maximumSumOfHeights(maxHeights: List<Int>): Long {
    val n = maxHeights.size
    val stack = Stack<Int>()

    val greaterRight = IntArray(n) { n }
    for (i in 0 until n) {
        while (stack.isNotEmpty() && maxHeights[i] >= maxHeights[stack.peek()]) {
            val start = stack.pop()
            greaterRight[start] = i
        }
        stack.push(i)
    }

    stack.clear()
    val smallerRight = IntArray(n) { n }
    for (i in 0 until n) {
        while (stack.isNotEmpty() && maxHeights[i] < maxHeights[stack.peek()]) {
            val start = stack.pop()
            smallerRight[start] = i
        }
        stack.push(i)
    }

    stack.clear()
    val greaterLeft = IntArray(n) { -1 }
    for (i in (n - 1) downTo 0) {
        while (stack.isNotEmpty() && maxHeights[i] > maxHeights[stack.peek()]) {
            val end = stack.pop()
            greaterLeft[end] = i
        }
        stack.push(i)
    }

    stack.clear()
    val smallerLeft = IntArray(n) { -1 }
    for (i in (n - 1) downTo 0) {
        while (stack.isNotEmpty() && maxHeights[i] <= maxHeights[stack.peek()]) {
            val end = stack.pop()
            smallerLeft[end] = i
        }
        stack.push(i)
    }
    return 0
}