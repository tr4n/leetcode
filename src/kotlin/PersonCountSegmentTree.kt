package org.example

import java.util.*

class PersonCountSegmentTree(size: Int) {
    private val n = size
    private val tree = IntArray(4 * n)

    fun update(index: Int, value: Int) {
        update(1, 0, n - 1, index, index, value)
    }

    private fun update(node: Int, l: Int, r: Int, ul: Int, ur: Int, value: Int) {
        if (l > ur || r < ul) return
        if (l == r) {
            tree[node] = value - 1
            return
        }

        val mid = (l + r) / 2
        update(2 * node, l, mid, ul, ur, value)
        update(2 * node + 1, mid + 1, r, ul, ur, value)
        tree[node] = tree[2 * node] + tree[2 * node + 1]
    }

    fun getSum(start: Int, end: Int): Int {
        return query(1, 0, n - 1, start, end)
    }


    private fun query(node: Int, l: Int, r: Int, ul: Int, ur: Int): Int {
        if (l > ur || r < ul) return 0

        if (ul <= l && r <= ur) {
            return tree[node]
        }
        val mid = (l + r) / 2
        val left = query(2 * node, l, mid, ul, ur)
        val right = query(2 * node + 1, mid + 1, r, ul, ur)
        return left + right
    }
}

fun canSeePersonsCount(heights: IntArray): IntArray {
    val n = heights.size
    val stack = Stack<Int>()
    val segmentTree = PersonCountSegmentTree(n)

 //   println(heights.indices.toList())
  //  println(heights.toList())

    val result = IntArray(n)

    for (i in 0 until n) {

        while (stack.isNotEmpty() && heights[i] >= heights[stack.peek()]) {
            val top = stack.pop()
            val hiddenCount = segmentTree.getSum(top + 1, i - 1)
            result[top] = i - top - hiddenCount
            segmentTree.update(top, result[top])
          //  hiddenCount += (result[top] - 1)

        }
        stack.push(i)
    }
 //   println(result.toList())
  //  println(stack.map { heights[it] })
    if (stack.isEmpty()) {
        return result
    }

    var hiddenCount = 0
    var lastIndex = n - 1
    for (i in (n - 1) downTo 0) {
        if (stack.isEmpty()) break
        if (stack.peek() == i) {
            val top = stack.pop()
            result[top] = lastIndex - top - hiddenCount
            hiddenCount = 0
            lastIndex = i
        } else {
            hiddenCount += (result[i] - 1)
        }
    }
    return result
}

fun main() {
    println(
        canSeePersonsCount(intArrayOf(17, 8, 14, 6, 16, 13, 1, 18, 11, 5)).toList()
    )
}
