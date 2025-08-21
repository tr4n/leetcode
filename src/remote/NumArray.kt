package remote

import kotlin.math.max

class NumArray(val nums: IntArray) {
    val n = nums.size
    val prefixSum = IntArray(n) { 0 }

    init {
        prefixSum[0] = nums[0]
        for (i in 1 until n) {
            prefixSum[i] = prefixSum[i - 1] + nums[i]
        }
    }


    fun sumRange(left: Int, right: Int): Int {
        if (left == right) return nums[left]
        if (left == 0) return prefixSum[right]
        return prefixSum[right] - prefixSum[left - 1]
    }
}

fun replaceElements(arr: IntArray): IntArray {
    val n = arr.size
    if (n == 1) return intArrayOf(-1)

    val result = IntArray(n) { -1 }
    var currentMax = arr.last()

    for (i in (n - 2) downTo 0) {
        result[i] = currentMax
        if (arr[i] > currentMax) {
            currentMax = arr[i]
        }
    }

    return result
}

fun subarraySum(nums: IntArray): Int {
    val n = nums.size
    val prefixSum = IntArray(n) { 0 }

    prefixSum[0] = nums[0]
    for (i in 1 until n) {
        prefixSum[i] = prefixSum[i - 1] + nums[i]
    }

    var total = 0

    for (i in 0 until n) {
        val start = max(0, i - nums[i])
        total += if (start == 0) {
            prefixSum[i]
        } else {
            prefixSum[i] - prefixSum[start - 1]
        }
    }

    return total
}