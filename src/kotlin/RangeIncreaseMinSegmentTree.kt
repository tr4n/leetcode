package org.example

import java.util.*
import kotlin.math.abs

class RangeIncreaseMinSegmentTree(private val data: IntArray) {
    private val n = data.size
    private val tree = IntArray(4 * n)
    private val lazy = IntArray(4 * n)

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
        tree[node] = minOf(tree[2 * node], tree[2 * node + 1])
    }

    private fun push(node: Int, l: Int, r: Int) {
        if (lazy[node] == 0) return
        tree[node] += lazy[node]
        if (l != r) {
            lazy[2 * node] += lazy[node]
            lazy[2 * node + 1] += lazy[node]
        }
        lazy[node] = 0
    }

    fun rangeIncreasement(ql: Int, qr: Int, value: Int) {
        update(1, 0, n - 1, ql, qr, value)
    }

    private fun update(node: Int, l: Int, r: Int, ql: Int, qr: Int, value: Int) {
        push(node, l, r)
        if (l > qr || r < ql) return
        if (ql <= l && r <= qr) {
            lazy[node] += value
            push(node, l, r)
            return
        }
        val mid = (l + r) / 2
        update(2 * node, l, mid, ql, qr, value)
        update(2 * node + 1, mid + 1, r, ql, qr, value)
        tree[node] = minOf(tree[2 * node], tree[2 * node + 1])
    }

    fun rangeMinQuery(ql: Int, qr: Int): Int {
        return query(1, 0, n - 1, ql, qr)
    }

    fun query(index: Int): Int {
        return query(1, 0, n - 1, index)
    }

    private fun query(node: Int, l: Int, r: Int, index: Int): Int {
        push(node, l, r)
        if (l == r) return tree[node]

        val mid = (l + r) / 2
        return if (index <= mid) {
            query(2 * node, l, mid, index)
        } else {
            query(2 * node + 1, mid + 1, r, index)
        }
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int): Int {
        push(node, l, r)
        if (l > qr || r < ql) return Int.MAX_VALUE
        if (ql <= l && r <= qr) return tree[node]

        val mid = (l + r) / 2
        val leftMin = query(2 * node, l, mid, ql, qr)
        val rightMin = query(2 * node + 1, mid + 1, r, ql, qr)
        return minOf(leftMin, rightMin)
    }
}

fun minNumberOperations(target: IntArray): Int {
    val n = target.size
    val segmentTree = RangeIncreaseMinSegmentTree(IntArray(n))

    val stack = Stack<Int>()

    val greaterRight = IntArray(n) { n }
    for (i in 0 until n) {
        while (stack.isNotEmpty() && target[i] > target[stack.peek()]) {
            val start = stack.pop()
            greaterRight[start] = i
        }
        stack.push(i)
    }

    stack.clear()
    val greaterLeft = IntArray(n) { -1 }
    for (i in (n - 1) downTo 0) {
        while (stack.isNotEmpty() && target[i] > target[stack.peek()]) {
            val end = stack.pop()
            greaterLeft[end] = i
        }
        stack.push(i)
    }

    stack.clear()
    val smallerRight = IntArray(n) { n }
    for (i in 0 until n) {
        while (stack.isNotEmpty() && target[i] < target[stack.peek()]) {
            val start = stack.pop()
            smallerRight[start] = i
        }
        stack.push(i)
    }

    stack.clear()
    val smallerLeft = IntArray(n) { -1 }
    for (i in (n - 1) downTo 0) {
        while (stack.isNotEmpty() && target[i] < target[stack.peek()]) {
            val end = stack.pop()
            smallerLeft[end] = i
        }
        stack.push(i)
    }

    val minRanges = target.withIndex()
        .sortedBy { (_, value) -> value }
        .map { (index, value) ->
            Triple(index, smallerLeft[index] + 1, smallerRight[index] - 1)
        }.distinctBy { it.second to it.third }

    //  println(minRanges)
    var operator = 0
    for ((index, start, end) in minRanges) {
        val currentValue = segmentTree.query(index)
        val diff = target[index] - currentValue
        operator += diff
        segmentTree.rangeIncreasement(start, end, diff)
    }

    return operator
}

fun minimumOperations(nums: IntArray, target: IntArray): Long {
    val n = nums.size
    val positiveNumbers = IntArray(n)
    val negativeNumbers = IntArray(n)
    for (i in 0 until nums.size) {
        val delta = target[i] - nums[i]
        if (delta >= 0) {
            positiveNumbers[i] = delta
        } else {
            negativeNumbers[i] = -delta
        }
    }
    var operator = 0L
    for (numbers in listOf(positiveNumbers, negativeNumbers)) {
        val segmentTree = RangeIncreaseMinSegmentTree(IntArray(n))
        val stack = Stack<Int>()
        val smallerRight = IntArray(n) { n }
        for (i in 0 until n) {
            while (stack.isNotEmpty() && numbers[i] < numbers[stack.peek()]) {
                val start = stack.pop()
                smallerRight[start] = i
            }
            stack.push(i)
        }

        stack.clear()
        val smallerLeft = IntArray(n) { -1 }
        for (i in (n - 1) downTo 0) {
            while (stack.isNotEmpty() && numbers[i] < numbers[stack.peek()]) {
                val end = stack.pop()
                smallerLeft[end] = i
            }
            stack.push(i)
        }

        val minRanges = numbers.withIndex()
            .sortedBy { (_, value) -> value }
            .mapNotNull { (index, value) ->
                val left = smallerLeft[index] + 1
                val right = smallerRight[index] - 1
                if (value == Int.MIN_VALUE || left > right) return@mapNotNull null
                Triple(index, left, right)
            }.distinctBy { it.second to it.third }

        //  println(numbers.indices.toList())
        //   println(numbers.toList())
        //   println(minRanges.toList())

        for ((index, start, end) in minRanges) {
            val currentValue = segmentTree.query(index)
            val diff = numbers[index] - currentValue
            operator += abs(diff).toLong()
            segmentTree.rangeIncreasement(start, end, diff)
            //   println(List(n) {segmentTree.query(it)})
        }
    }
    return operator
}


fun corpFlightBookings(bookings: Array<IntArray>, n: Int): IntArray {
    val tree = RangeIncreaseMinSegmentTree(IntArray(n))

    for ((first, last, seats) in bookings) {
        tree.rangeIncreasement(first - 1, last - 1, seats)
    }
    return IntArray(n) {
        tree.query(it)
    }
}

fun main() {
    println(
        minimumOperations(
            intArrayOf(1, 3, 2),
            intArrayOf(2, 1, 4)
        )
    )
}