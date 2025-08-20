package org.example

import java.util.*
import kotlin.math.abs


class ListNode(var `val`: Int) {
    var next: ListNode? = null
}


fun main() {
    println(
        hasSpecialSubstring("ii", 1)
    )
}

fun hasSpecialSubstring(s: String, k: Int): Boolean {
    val n = s.length
    if (n == 1) return k == 1
    var i = 0
    while (i <= n - k) {
        var j = i + 1
        var notMatch = false
        while (j < n && j < i + k) {
            if (s[j] != s[i]) {
                i = j
                notMatch = true
                break
            }
            j++
        }
        if (notMatch) continue
        if ((j == n || s[j] != s[i]) && (i == 0 || s[i - 1] != s[i])) {
            return true
        }
        i = j
    }
    return false
}

fun canBeTypedWords(text: String, brokenLetters: String): Int {
    val broken = brokenLetters.toSet()
    return text.split(" ").count { word ->
        word.none { it in broken }
    }
}

fun removeKdigits(num: String, k: Int): String {
    var number = num
    var count = k
    if (number.isEmpty() || count >= number.length) return "0"

    val stack = Stack<Char>()
    for (digit in number) {
        while (stack.isNotEmpty() && count > 0 && stack.peek() > digit) {
            stack.pop()
            count--
        }
        stack.push(digit)
    }

    while (count > 0 && stack.isNotEmpty()) {
        stack.pop()
        count--
    }

    val result = StringBuilder()
    while (stack.isNotEmpty()) {
        result.append(stack.pop())
    }

    number = result.toString().reversed().trimStart('0')

    return number.ifEmpty { "0" }
}

fun removeDigit(number: String, digit: Char): String {
    val n = number.length
    var maxNum = ""
    for (i in 0 until n) {
        if (number[i] == digit) {
            val num = StringBuilder()
            if (i > 0) num.append(number.substring(0, i))
            if (i < n - 1) num.append(number.substring(i + 1, n))
            val newNumber = num.toString()

            if (maxNum.isEmpty()) {
                maxNum = newNumber
                continue
            }
            // println("$newNumber - $maxNum")
            for (i in 0 until newNumber.length) {
                if (newNumber[i] == maxNum[i]) continue
                if (newNumber[i] > maxNum[i]) {
                    maxNum = newNumber
                }
                break
            }
        }
    }
    return maxNum
}

fun secondHighest(s: String): Int {
    var first = -1
    var second = -1


    for (c in s) {
        val num = c.digitToIntOrNull() ?: continue
        if (num == first || num == second) continue

        if (num > first) {
            second = first
            first = num
            continue
        }

        if (num > second) {
            second = num
        }
    }
    return second
}


fun topStudents(
    positive_feedback: Array<String>,
    negative_feedback: Array<String>,
    report: Array<String>,
    student_id: IntArray,
    k: Int
): List<Int> {
    val n = report.size
    val positiveFeedbacks = positive_feedback.toSet()
    val negativeFeedbacks = negative_feedback.toSet()
    val students = mutableListOf<Pair<Int, Int>>()

    for (i in 0 until n) {
        var score = 0
        for (word in report[i].split(" ")) {
            if (word in positiveFeedbacks) score += 3
            else if (word in negativeFeedbacks) score--
        }
        students.add(student_id[i] to score)
    }
    students.sortWith(compareBy({ -it.second }, { it.first }))
    return students.take(k).map { it.first }

}

fun reconstructQueue(people: Array<IntArray>): Array<IntArray> {
    people.sortWith(compareBy({ -it[0] }, { it[1] }))
    val list = mutableListOf<IntArray>()

    for (item in people) {
        val (_, k) = item
        list.add(k, item)
    }
    return list.toTypedArray()
}


fun countMaxOrSubsets(nums: IntArray): Int {
    val n = nums.size
    var count = 0
    var maxOr = -1
    for (mask in 0 until (1 shl n)) {
        var result = 0
        //  val set = mutableListOf<Int>()
        for (i in 0 until n) {
            if ((mask shr i) and 1 == 1) {
                result = result or nums[i]
                //   set.add(nums[i])
            }
        }
        if (result == maxOr) {
            count++
            //   println("${count} ${maxOr}   :  ${set}")
        }
        if (result > maxOr) {
            count = 1
            maxOr = result
            //    println("${count} ${maxOr}   :  ${set}")
        }
    }
    return count
}

fun countBits(n: Int): IntArray {
    return IntArray(n + 1) { it.countOneBits() }
}

fun hammingWeight(n: Int): Int {
    var count = 0
    for (i in 0..31) {
        if (n and (1 shl i) != 0) count++
    }
    return count
}

fun reverse(x: Int): Int {
    val xLong = x.toLong()
    val num = if (xLong >= 0) xLong.toString().reversed().toLong() else -abs(xLong).toString().reversed().toLong()
    return if (num in Int.MIN_VALUE.toLong()..Int.MAX_VALUE.toLong()) num.toInt() else 0
}

fun twoSum(numbers: IntArray, target: Int): IntArray {
    val targetLong = target.toLong()
    var left = 0
    var right = numbers.size - 1

    while (left < right) {
        val second = numbers[left]
        val third = numbers[right]
        val sum = second.toLong() + third
        when {
            sum == targetLong -> {
                return intArrayOf(left + 1, right + 1)
            }

            sum > targetLong -> right--

            else -> left++
        }
    }
    return intArrayOf()
}

fun fourSum(nums: IntArray, target: Int): List<List<Int>> {
    val sortedNums = nums.sorted()
    val targetLong = target.toLong()
    val result = mutableSetOf<List<Int>>()
    for (i in 0 until sortedNums.size - 1) {
        for (j in i + 1 until sortedNums.size) {
            val first = sortedNums[i]
            val second = sortedNums[j]
            var left = 0
            var right = sortedNums.size - 1

            while (left < right) {
                if (left == i || left == j) {
                    left++
                    continue
                }
                if (right == i || right == j) {
                    right--
                    continue
                }
                val third = sortedNums[left]
                val forth = sortedNums[right]

                val sum: Long = first.toLong() + second + third + forth
                when {
                    sum == targetLong -> {
                        result.add(listOf(first, second, third, forth).sorted())
                        right--
                    }

                    sum > targetLong -> right--

                    else -> left++
                }
            }
        }
    }
    return result.map { it.toList() }
}


fun threeSum(nums: IntArray): List<List<Int>> {
    val sortedNums = nums.sorted()
    val result = mutableSetOf<List<Int>>()
    for (i in sortedNums.indices) {
        val first = sortedNums[i]
        var left = 0
        var right = sortedNums.size - 1

        while (left < right) {
            if (left == i) left++
            if (right == i) right--
            if (left >= right) break

            val second = sortedNums[left]
            val third = sortedNums[right]
            when {
                second + third == -first -> {
                    result.add(listOf(first, second, third).sorted())
                    right--
                }

                second + third > -first -> right--

                second + third < -first -> left++
            }
        }

    }
    return result.map { it.toList() }
}

fun threeSumClosest(nums: IntArray, target: Int): Int {
    val sortedNums = nums.sorted()
    var sum = Int.MAX_VALUE
    var delta = Int.MAX_VALUE
    for (i in sortedNums.indices) {
        val first = sortedNums[i]
        var left = 0
        var right = sortedNums.size - 1

        while (left < right) {
            if (left == i) left++
            if (right == i) right--
            if (left >= right) break

            val second = sortedNums[left]
            val third = sortedNums[right]
            val tripleSum = first + second + third

            if (tripleSum == target) {
                return tripleSum
            }

            val tripleDelta = abs(tripleSum - target)

            if (tripleDelta < delta) {
                delta = tripleDelta
                sum = tripleSum
            }

            when {
                second + third > target - first -> right--
                second + third < target - first -> left++
            }
        }

    }
    return sum
}
