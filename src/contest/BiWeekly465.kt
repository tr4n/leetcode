package contest

import kotlin.math.abs
import kotlin.math.sqrt

fun maxProduct(nums: IntArray): Long {
    class Node {
        val children = arrayOfNulls<Node>(2)
        var maxValue = 0L
    }

    val root = Node()

    for (num in nums) {
        var node = root
        node.maxValue = maxOf(node.maxValue, num.toLong())
        for (bit in 31 downTo 0) {
            val b = (num shr bit) and 1
            if (node.children[b] == null) node.children[b] = Node()
            node = node.children[b]!!
            node.maxValue = maxOf(node.maxValue, num.toLong())
        }
    }
    var maxProduct = 0L
    for (first in nums) {
        var node = root
        var notFound = false
        for (bit in 31 downTo 0) {
            val firstBit = 1 and (first shr bit)
            val child0 = node.children[0]
            val child1 = node.children[1]
            when {
                firstBit == 1 -> {
                    if (child0 == null) {
                        notFound = true
                        break
                    }
                    node = child0
                }

                child1 != null -> {
                    node = child1
                }

                child0 != null -> {
                    node = child0
                }

                else -> {
                    notFound = true
                    break
                }
            }
        }
        if (notFound) continue
        val second = node.maxValue
        maxProduct = maxOf(maxProduct, second * first)
    }

    return maxProduct
}

fun minDifference(n: Int, k: Int): IntArray {
    val divisors = mutableListOf(1)
    for (i in 2..n / 2) {
        if (n % i == 0) divisors.add(i)
    }

    val path = mutableListOf<Int>()
    var delta = Int.MAX_VALUE
    var result = listOf<Int>()
  //  println(divisors)
    fun dfs(num: Int, min: Int, maxDivisor: Int) {
        if (num < 1) return
        if (path.size > k) return

        if (num == 1) {
            val minDivisor = if(path.size < k) 1 else min
            if (maxDivisor - minDivisor >= delta) return
          //  println(path)
            result = path.toList()
            delta = maxDivisor - min
            return
        }

        for (i in divisors.indices) {
            val divisor = divisors[i]
            if (num % divisor != 0) continue
            path.add(divisor)
            dfs(num / divisor, minOf(min, divisor), maxOf(maxDivisor, divisor))
            path.removeLast()
        }
    }
    dfs(n, Int.MAX_VALUE, Int.MIN_VALUE)
    return IntArray(k) {
        result.getOrNull(it) ?: 1
    }
}

fun main(){
    println(
        minDifference(100, 2).toList()
    )

}