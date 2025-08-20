package org.example

import java.util.*


class SecondGreaterElement(private val data: IntArray) {
    private val n = data.size
    private val tree = IntArray(4 * n)

    init {
        build(1, 0, n - 1)
    }


    fun findIndex(value: Int, start: Int, end: Int): Int {
        return query(1, 0, n - 1, start, end, value)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = l
            return
        }

        val mid = (l + r) / 2
        build(2 * node, l, mid)
        build(2 * node + 1, mid + 1, r)

        val leftNode = tree[2 * node]
        val rightNode = tree[2 * node + 1]
        tree[node] = when {
            data[leftNode] > data[rightNode] -> leftNode
            data[leftNode] < data[rightNode] -> rightNode
            else -> minOf(leftNode, rightNode)
        }
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int, value: Int): Int {
        if (r < ql || l > qr) return -1

        val index = tree[node]
        if (ql <= l && r <= qr && data[index] < value) return -1

        if (l == r) return if (data[index] > value) index else -1

        val mid = (l + r) / 2
        val leftResult = query(2 * node, l, mid, ql, qr, value)
        return if (leftResult != -1) leftResult
        else query(2 * node + 1, mid + 1, r, ql, qr, value)
    }
}

fun secondGreaterElement(nums: IntArray): IntArray {
    val n = nums.size
    if (n < 3) return IntArray(n) { -1 }
    val stack = Stack<Int>()
    val tree = SecondGreaterElement(nums)

    val greaterRight = IntArray(n) { n }
    for (i in 0 until n) {
        while (stack.isNotEmpty() && nums[i] > nums[stack.peek()]) {
            val start = stack.pop()
            greaterRight[start] = i
        }
        stack.push(i)
    }

    //  println(greaterRight.toList())
    val secondGreater = IntArray(n) { -1 }
    for (i in 0 until n) {
        if (i > 0 && nums[i] == nums[i - 1]) {
            secondGreater[i] = secondGreater[i - 1]
            continue
        }
        val start = greaterRight[i]
        if (start >= n) continue
        val index = tree.findIndex(nums[i], start + 1, n - 1)
        if (index < 0) continue
        secondGreater[i] = nums[index]
    }

    return secondGreater
}

fun nextGreaterElement(nums1: IntArray, nums2: IntArray): IntArray {
    val n = nums1.size
    val numToIndex = mutableMapOf<Int, Int>()
    for ((i, num) in nums2.withIndex()) {
        numToIndex[num] = i
    }

    val tree = SecondGreaterElement(nums2)
    val result = IntArray(n) { -1 }
    for ((i, num) in nums1.withIndex()) {
        val j = numToIndex[num] ?: continue
        val index = tree.findIndex(num, j + 1, nums2.size - 1)
        if (index < 0) continue
        result[i] = nums2[index]
    }

    return result

}

fun main() {
    println(
        secondGreaterElement(intArrayOf(2, 4, 0, 9, 6)).toList()
    )
}