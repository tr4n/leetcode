package contest

import java.util.*
import kotlin.math.abs
import kotlin.math.ceil

class Weekly71 {

    fun numRabbits(answers: IntArray): Int {
        return answers.toList()
            .groupingBy { it }
            .eachCount()
            .toList()
            .sumOf { (answer, count) ->

                (1 + answer) * ceil(count.toDouble() / (1.0 + answer)).toInt()
            }
    }

    fun reachingPoints(sx: Int, sy: Int, tx: Int, ty: Int): Boolean {

        val pq =
            PriorityQueue<Pair<Long, Long>>(compareBy { abs(it.first - tx.toLong()) + abs(it.second - ty.toLong()) })
        pq.add(sx.toLong() to sy.toLong())
        val targetSum = (sx.toLong() + sy.toLong()).coerceAtLeast(1000)

        val visited = mutableSetOf<Pair<Long, Long>>()
        val targetPair = tx.toLong() to ty.toLong()
        while (pq.isNotEmpty()) {
            val pair = pq.poll()
            if (pair == targetPair) {
                return true
            }
            if (pair in visited) continue
            visited.add(pair)
            val (x, y) = pair
            val sum = x + y
            if (sum <= targetSum) pq.add(sum to y)
            if (sum <= targetSum) pq.add(x to sum)
        }
        return false
    }

    fun maxXorSubsequences(nums: IntArray): Int {
        fun maxSubsequenceXor(arr: IntArray): Int {
            val basis = IntArray(32)
            for (num in arr) {
                var x = num
                for (i in 31 downTo 0) {
                    if (((x shr i) and 1) == 0) continue
                    if (basis[i] == 0) {
                        basis[i] = x
                        break
                    }
                    x = x xor basis[i]
                }
            }

            var res = 0
            for (i in 31 downTo 0) {
                res = maxOf(res, res xor basis[i])
            }
            return res
        }

        val arr = nums + nums
        return maxSubsequenceXor(arr)
    }
}

fun main() {
    val contest = Weekly71()

    println(contest.numRabbits(intArrayOf(1, 1, 2, 2, 2, 3, 3)))
}