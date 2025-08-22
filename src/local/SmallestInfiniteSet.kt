package local

import java.util.TreeSet

class SmallestInfiniteSet() {
    private val treeSet = TreeSet<Int>()

    init {
        treeSet.addAll(1..1100)
    }

    fun popSmallest(): Int {
        return treeSet.pollFirst() ?: 0
    }

    fun addBack(num: Int) {
        treeSet.add(num)
    }

}

fun firstMissingPositive(nums: IntArray): Int {
    val n = nums.size
    var i = 0
    while (i < n) {
        val num = nums[i]
        val correctPos = num - 1
        if (num !in 1..n || i == correctPos) {
            i++
            continue
        }

        val tmp = nums[i]
        nums[correctPos] = tmp
        nums[i] = tmp
        i++
    }

    for(j in 0 until n) {
        if(nums[j] !=  j + 1) return j + 1
    }
    return n + 1
}