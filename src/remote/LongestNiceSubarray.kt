package remote

import java.util.*
import kotlin.math.abs

class LongestNiceSubarray(private val data: IntArray) {
    private val n = data.size
    private val tree = IntArray(4 * n)

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = data[l]
            return
        }
        val mid = (l + r) / 2
        build(2 * node, l, mid)
        build(2 * node + 1, mid + 1, r)
        val leftNode = tree[2 * node]
        val rightNode = tree[2 * node + 1]
        tree[node] = leftNode and rightNode
    }

    fun query(start: Int, end: Int): Int {
        val node = query(1, 0, n - 1, start, end)
        return node
    }


    private fun query(node: Int, l: Int, r: Int, ul: Int, ur: Int): Int {
        if (l > ur || r < ul) return -1

        if (ul <= l && r <= ur) {
            return tree[node]
        }
        val mid = (l + r) / 2
        val left = query(2 * node, l, mid, ul, ur)
        val right = query(2 * node + 1, mid + 1, r, ul, ur)
        return left and right
    }
}

fun longestNiceSubarray(nums: IntArray): Int {
    val n = nums.size
    val queue = ArrayDeque<Int>()
    var totalMax = 1
    var currentMax = 1
    var currentXor = nums[0]
    var currentSum = nums[0]
    queue.add(nums[0])

    for (i in 1 until n) {
        val num = nums[i]
        while (queue.isNotEmpty() && currentXor xor num != currentSum + num) {
            val leftNum = queue.removeFirst()
            currentMax--
            currentXor = currentXor xor leftNum
            currentSum -= leftNum
        }

        currentMax++
        currentXor = currentXor xor num
        currentSum = currentSum + num
        queue.add(num)
        totalMax = maxOf(totalMax, currentMax)
    }

    return totalMax
}

fun longestOnes(nums: IntArray, k: Int): Int {
    val prefixMap = TreeMap<Int, Int>()
    prefixMap[-1] = -1

    var sum = 0
    var maxLength = 0

    for (i in nums.indices) {
        sum += nums[i]
        val iSum = i - sum

        prefixMap[iSum] = minOf(prefixMap[iSum] ?: i, i)

        val entry = prefixMap.ceilingEntry(iSum - k)
        if (entry != null) {
            val left = entry.value
            maxLength = maxOf(maxLength, i - left)
        }
    }

    return maxLength
}

fun equalSubstring(s: String, t: String, maxCost: Int): Int {
    val n = s.length
    val costs = IntArray(n) { abs(s[it] - t[it]) }
    val treeMap = TreeMap<Int, Int>()
    treeMap[0] = -1

    var sum = 0
    var maxLength = 0
    for (i in 0 until n) {
        sum += costs[i]
        val entry = treeMap.ceilingEntry(sum - maxCost)
        treeMap[sum] = minOf(treeMap[sum] ?: i, i)
        if (entry != null) {
            val left = entry.value
            maxLength = maxOf(maxLength, i - left)
        }
    }
    return maxLength
}

fun minSubArrayLen(target: Int, nums: IntArray): Int {
    val n = nums.size
    val treeMap = TreeMap<Int, Int>()
    treeMap[0] = -1

    var sum = 0
    var minLength = 2 * n
    for (i in 0 until n) {
        sum += nums[i]
        val entry = treeMap.floorEntry(sum - target)
        treeMap[sum] = maxOf(treeMap[sum] ?: i, i)
        if (entry != null) {
            val left = entry.value
            minLength = minOf(minLength, i - left)
        }
    }
    return if (minLength > n) 0 else minLength
}

fun findLength(nums1: IntArray, nums2: IntArray): Int {
    val m = nums1.size
    val n = nums2.size

    val dp = Array(m) { IntArray(n) }

    for (j in 0 until n) dp[0][j] = if (nums1[0] == nums2[j]) 1 else 0
    for (i in 0 until m) dp[i][0] = if (nums2[0] == nums1[i]) 1 else 0

    for (i in 1 until m) {
        for (j in 1 until n) {
            dp[i][j] = if (nums1[i] == nums2[j]) {
                dp[i - 1][j - 1] + 1
            } else {
                //   minOf(dp[i - 1][j], dp[i][j - 1])
                0
            }
        }
    }
    //  println(dp.joinToString("\n") { it.toList().toString() })
    var maxLength = 0
    for (i in 0 until m) {
        for (j in 0 until n) {
            maxLength = maxOf(maxLength, dp[i][j])
        }
    }
    return maxLength
}

fun getSum(a: Int, b: Int): Int {
    var x = a
    var y = b
    // Xor  Cộng không nhớ
    // And  Trả về các bit nhớ
    // And dịch trái 1 bit  + xor sẽ thành Cộng có nhớ
    // a + b = a xor b + 2 * (a and b)
    while (y != 0) {
        val carry = (x and y) shl 1
        x = x xor y
        y = carry
    }
    return x
}

fun longestBeautifulSubstring(word: String): Int {
    val n = word.length
    val vowels = setOf('u', 'e', 'o', 'a', 'i')


    var totalMax = 0
    var currentMax = 0

    var lastCharacter = ('a'.code - 1).toChar()
    val set = mutableSetOf<Char>()

    for (i in 0 until n) {
        val c = word[i]

        if (set.isEmpty() && c != 'a') continue

        if (c.code >= lastCharacter.code) {
            currentMax++
            lastCharacter = c
            set.add(c)
        } else {
            if (set.containsAll(vowels)) {
                totalMax = maxOf(totalMax, currentMax)
            }
            set.clear()
            if (c == 'a') {
                set.add(c)
                currentMax = 1
                lastCharacter = c
            } else {
                lastCharacter = ('a'.code - 1).toChar()
                currentMax = 0
            }
        }
    }

    return if (set.containsAll(vowels)) {
        maxOf(totalMax, currentMax)
    } else totalMax
}

fun maxConsecutiveAnswers(answerKey: String, k: Int): Int {
    fun findLongest(nums: List<Int>, k: Int): Int {
        val prefixMap = TreeMap<Int, Int>()
        prefixMap[-1] = -1

        var sum = 0
        var maxLength = 0

        for (i in nums.indices) {
            sum += nums[i]
            val iSum = i - sum

            prefixMap[iSum] = minOf(prefixMap[iSum] ?: i, i)

            val entry = prefixMap.ceilingEntry(iSum - k)
            if (entry != null) {
                val left = entry.value
                maxLength = maxOf(maxLength, i - left)
            }
        }

        return maxLength
    }

    var nums = answerKey.map { if (it == 'T') 0 else 1 }
    val longestFalse = findLongest(nums, k)
    nums = answerKey.map { if (it == 'T') 1 else 0 }
    val longestTrue = findLongest(nums, k)
    return maxOf(longestFalse, longestTrue)
}

fun maximumSubarraySum(nums: IntArray, k: Int): Long {
    val n = nums.size

    if (n < k) return 0L

    var sum = 0L
    val set = mutableSetOf<Int>()
    val subArray = ArrayDeque<Int>()
    var maxSum = 0L

    for (i in 0 until n) {
        val num = nums[i]
        while (num in set) {
            val firstNum = subArray.removeFirst()
            sum -= firstNum.toLong()
            set.remove(firstNum)
        }

        set.add(num)
        sum += num.toLong()
        subArray.addLast(num)
        if (set.size == k) {
            maxSum = maxOf(sum, maxSum)
            val firstNum = subArray.removeFirst()
            sum -= firstNum.toLong()
            set.remove(firstNum)
        }
    }
    return maxSum

}

fun findMaxConsecutiveOnes(nums: IntArray): Int {
    val n = nums.size
    var left = -1

    var maxLength = 0
    for (i in 0..n) {
        if (i == n || nums[i] == 0) {
            maxLength = maxOf(maxLength, i - left - 1)
            left = i
        }
    }
    return maxLength
}

fun lengthOfLIS(nums: IntArray, k: Int): Int {
    val n = nums.size

    val maxValue = nums.max()

    val tree = MaxSegmentTree(IntArray(maxValue + 1))
    var maxLength = 0
    for (i in 0 until n) {
        val num = nums[i]
        val left = (num - k).coerceAtLeast(0)
        val right = num - 1
        if (left > right) continue
        val d = tree.query(left, right) + 1
        maxLength = maxOf(maxLength, d)
        tree.update(num, d)
    }
//    println(dp.joinToString ("\n"){ it.toList().toString() })
    return maxLength
}

fun main() {
    println(
        lengthOfLIS(
            intArrayOf(1, 100, 500, 100000, 100000),
            100000
        )
    )
}