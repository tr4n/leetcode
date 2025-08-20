package org.example

import java.math.BigInteger
import kotlin.math.*

fun IntArray.print() = toList().toString()

fun Array<IntArray>.print() = joinToString("\n") { it.print() }

fun parseStringToIntArray2D(input: String): Array<IntArray> {
    return input
        .trim() // loại bỏ khoảng trắng đầu/cuối nếu có
        .removePrefix("[") // bỏ dấu `[` bên ngoài
        .removeSuffix("]") // bỏ dấu `]` bên ngoài
        .split("],") // tách từng dòng, lưu ý dấu `],` giữ lại `]` cuối cùng dòng
        .map { row ->
            row.replace("[", "") // bỏ dấu `[` bên trong
                .replace("]", "") // bỏ dấu `]` bên trong
                .split(",") // tách các phần tử
                .map { it.trim().toInt() } // chuyển sang Int
                .toIntArray()
        }
        .toTypedArray()
}

fun main() {
    println("=================")
    println(
        canCross(intArrayOf(0, 1, 3, 5, 6, 8, 12, 17))
    )
}

fun countDistinctIntegers(nums: IntArray): Int {
    val set = mutableSetOf<Long>()
    for(num in nums) {
        val other = num.toString().reversed().toLong()
        set.add(num.toLong())
        set.add(other)
    }
    return set.sum().toInt()
}

fun canCross(stones: IntArray): Boolean {
    val n = stones.size
    if (n == 2) return stones[0] == 0 && stones[1] == 1
    if (stones[0] != 0 || stones[1] != 1) {
        return false
    }
    val stoneIndex = mutableMapOf<Int, Int>()
    for (i in 0 until n) {
        stoneIndex[stones[i]] = i
    }
    val dp = Array(n) { mutableSetOf<Int>() }

    dp[0].add(0)
    dp[1].add(1)

    for (i in 1 until n) {
        for (jump in dp[i]) {
            val nextJumps = listOf(jump + 1, jump, jump - 1)
            for (j in nextJumps) {
                val index = stoneIndex[stones[i] + j]
                if (j > 0 && index != null) {
                    dp[index].add(j)
                }
            }
        }
    }

    // println(dp.joinToString() { it.size.toString()  })
    return dp[n - 1].isNotEmpty()
}


fun canCross1(stones: IntArray): Boolean {
    val n = stones.size
    if (n == 2) return stones[0] == 0 && stones[1] == 1

    val stoneSet = stones.toSet()
    val visitedSet = mutableSetOf<Pair<Int, Int>>()
    val queue = ArrayDeque<Pair<Int, Int>>()
    queue.add(1 to 1)

    while (queue.isNotEmpty()) {
        val (u, jump) = queue.removeFirst()
        visitedSet.add(u to jump)
        // println("visit: $u")
        if (u == stones.last()) return true
        val nextJumps = listOf(jump + 1, jump, jump - 1)

        for (nextJump in nextJumps) {
            val v = u + nextJump
            if (v to nextJump in visitedSet) continue
            if (nextJump > 0 && v in stoneSet) {
                queue.add(v to nextJump)
            }
        }
    }

//    val visitedState = mutableSetOf<Int>()
//    var finished = false
//    fun dfs(i: Int, jump: Int, visitedList: MutableSet<Int>): Boolean {
//        if (finished) return true
//        if (i == stones.last()) {
//            finished = true
//            return true
//        }
//
//        visitedList.add(i)
//
//        val nextJumps = listOf(jump + 1, jump, jump - 1)
//
//        for (nextJump in nextJumps) {
//            val u = i + nextJump
//            if (u in visitedState) continue
//            if (u in stoneSet && nextJump > 0) {
//                dfs(u, nextJump, visitedList)
//            }
//        }
//        return finished
//    }


    return false
}

fun maxResult(nums: IntArray, k: Int): Int {
    val n = nums.size
    if (n == 1) return nums[0]

    val dp = IntArray(n)
    val deque = ArrayDeque<Int>()

    dp[0] = nums[0]
    deque.add(0)

    for (i in 1 until n) {
        dp[i] = nums[i] + dp[deque.first()]

        if (deque.isNotEmpty() && deque.first() <= i - k) {
            deque.removeFirst()
        }
        while (deque.isNotEmpty() && dp[deque.last()] < dp[i]) deque.removeLast()
        deque.addLast(i)
    }
    //  println(dp.toList())
    return dp[n - 1]
}

fun minJumps(nums: IntArray): Int {
    if (nums.size == 1) return 0
    if (nums.size == 2) return 1
    if (nums.first() == nums.last()) return 1

    val arr = mutableListOf<Int>()
    for (i in 0 until nums.size) {
        if (i < 2 || nums[i] != nums[i - 1] || nums[i] != nums[i - 2]) {
            arr.add(nums[i])
        }
    }
    val n = arr.size

    val valueToIndices = mutableMapOf<Int, MutableList<Int>>()

    for (i in arr.indices) {
        if (valueToIndices[arr[i]] == null) {
            valueToIndices[arr[i]] = mutableListOf(i)
        } else valueToIndices[arr[i]]?.add(i)
    }

    val visited = BooleanArray(n)
    val queue = ArrayDeque<Int>()
    queue.add(0)
    var steps = 0
    while (queue.isNotEmpty()) {
        repeat(queue.size) {
            val u = queue.removeFirst()
            if (u == n - 1) return steps

            if (u - 1 >= 0 && !visited[u - 1]) {
                visited[u - 1] = true
                queue.add(u - 1)
            }
            if (u + 1 < n && !visited[u + 1]) {
                visited[u + 1] = true
                queue.add(u + 1)
            }

            val neighbors = valueToIndices[arr[u]] ?: emptyList()
            for (v in neighbors) {
                if (!visited[v]) {
                    visited[v] = true
                    queue.add(v)
                }
            }

            valueToIndices.remove(arr[u])
        }
        steps++
    }

    return -1
}


fun maxSum(nums: IntArray): Int {
    val maxNum = nums.max()
    if (maxNum < 0) return maxNum
    val numbers = nums.filter { it >= 0 }.distinct()
    var currentSum = numbers[0]
    var maxSum = numbers[0]

    for (i in 1 until numbers.size) {
        currentSum = maxOf(numbers[i], currentSum + numbers[i])
        maxSum = maxOf(maxSum, currentSum)
    }

    return maxSum
}

fun canReach(s: String, minJump: Int, maxJump: Int): Boolean {
    if (s.last() != '0') return false
    val n = s.length
    val count = IntArray(n + 1)

    for (i in 0 until n) {
        count[i + 1] = if (s[i] == '0') {
            count[i] + 1
        } else count[i]
    }
    println(count.toList())
    var left = 1
    var right = 1
    while (right < n && left < n) {
        left = left + minJump
        right = min(left + maxJump, n)
        val cnt = count[right] - count[left - 1]
        println("$left $right $cnt")
        if (right == n) return true
        if (cnt <= 0) {
            return false
        }
    }
    return false
}

fun canReach(arr: IntArray, start: Int): Boolean {
    val n = arr.size
    var result = false

    fun fill(arr: IntArray, index: Int, visited: BooleanArray) {
        if (index !in arr.indices || visited[index]) return
        if (arr[index] == 0 || result) {
            result = true
            return
        }
        visited[index] = true
        fill(arr, index - arr[index], visited)
        fill(arr, index + arr[index], visited)
    }
    fill(arr, start, BooleanArray(n))
    return result
}

fun canJump(nums: IntArray): Boolean {
    val n = nums.size
    val dp = IntArray(n)

    for (i in 1 until n) {
        dp[i] = 1_000_000_000
        for (j in 0 until i) {
            val maxJump = nums[j]
            if (j + maxJump < i) continue
            dp[i] = min(dp[i], dp[j] + 1)
        }
    }
    return dp[n - 1] < n
}

fun jump(nums: IntArray): Int {
    val n = nums.size
    val dp = IntArray(n)

    for (i in 1 until n) {
        dp[i] = 1_000_000_000
        for (j in 0 until i) {
            val maxJump = nums[j]
            if (j + maxJump < i) continue
            dp[i] = min(dp[i], dp[j] + 1)
        }
    }
    return dp[n - 1]
}

fun divisorGame(n: Int): Boolean {
    if (n == 1) return false
    if (n == 2) return true
    if (n == 3) return false
    if (n == 4) return true

    val winSet = mutableSetOf(2, 4)

    for (i in 5..n) {
        for (j in 1..(i / 2)) {
            if (i % j == 0 && (i - j) !in winSet) {
                winSet.add(i)
                break
            }
        }
    }

    return n in winSet
}

fun productExceptSelf(nums: IntArray): IntArray {
    val n = nums.size
    val mLeft = IntArray(n + 2) { 1 }
    val mRight = IntArray(n + 2) { 1 }

    for (i in 1..n) {
        val j = n + 1 - i
        mLeft[i] = mLeft[i - 1] * nums[i - 1]
        mRight[j] = mRight[j + 1] * nums[j - 1]
    }

    return IntArray(n) {
        mLeft[it] * mRight[it + 2]
    }
}

fun longestUnivaluePath(root: TreeNode?): Int {
    root ?: return 0
    fun dfs(node: TreeNode?): Pair<Int, Int> {
        val emptyValue = 0 to 0
        if (node == null) return emptyValue

        val (lengthLeft, maxLeft) = dfs(node.left)
        val (lengthRight, maxRight) = dfs(node.right)

        val longestLeft = if (node.`val` == node.left?.`val`) 1 + lengthLeft else 1
        val longestRight = if (node.`val` == node.right?.`val`) 1 + lengthRight else 1
        val length = max(longestLeft, longestRight)

        var max = longestLeft + longestRight - 1
        max = max(max, maxLeft)
        max = max(max, maxRight)

        return length to max
    }
    return dfs(root).second - 1
}

fun pathSum(root: TreeNode?, targetSum: Int): List<List<Int>> {
    val result = mutableListOf<List<Int>>()
    fun visit(node: TreeNode?, path: MutableList<Int>) {
        node ?: return
        path.add(node.`val`)
        val children = listOfNotNull(node.left, node.right)
        val sum = path.sum()
        if (children.isEmpty() && sum == targetSum) {
            result.add(path.toMutableList())
            return
        }

        for (childNode in children) {
            visit(childNode, path)
            path.removeLast()
        }
    }
    visit(root, mutableListOf())
    return result
}

fun binaryTreePaths(root: TreeNode?): List<String> {
    val result = mutableListOf<String>()
    fun visit(node: TreeNode?, path: MutableList<Int>) {
        node ?: return
        path.add(node.`val`)
        val children = listOfNotNull(node.left, node.right)
        if (children.isEmpty()) {
            val str = path.joinToString("->") { it.toString() }
            result.add(str)
            return
        }

        for (childNode in children) {
            visit(childNode, path)
            path.removeLast()
        }
    }
    visit(root, mutableListOf())
    return result
}

fun smallestFromLeaf(root: TreeNode?): String {
    var smallestStr = ""
    fun visit(node: TreeNode?, visited: MutableSet<TreeNode>, list: MutableList<Int>) {
        node ?: return
        visited.add(node)
        list.add(0, node.`val`)
        val children = listOfNotNull(node.left, node.right)
        if (children.isEmpty()) {
            val str = list.joinToString("") { ('a'.code + it).toChar().toString() }
            if (smallestStr.isEmpty() || smallestStr > str) {
                smallestStr = str
            }
            return
        }


        for (childNode in children) {
            if (childNode !in visited) {
                visit(childNode, visited, list)
                list.removeFirstOrNull()
                visited.remove(childNode)
            }
        }
    }
    visit(root, mutableSetOf(), mutableListOf())
    return smallestStr
}

fun sumNumbers(root: TreeNode?): Int {
    var totalSum = 0L
    fun visit(node: TreeNode?, visited: MutableSet<TreeNode>, num: Long) {
        node ?: return
        visited.add(node)
        val newNum = num * 10 + node.`val`
        val children = listOfNotNull(node.left, node.right)
        if (children.isEmpty()) {
            totalSum += newNum
            return
        }


        for (childNode in children) {
            if (childNode !in visited) {
                visit(childNode, visited, newNum)
                visited.remove(childNode)
            }
        }
    }
    visit(root, mutableSetOf(), 0L)
    return totalSum.toInt()
}

fun maxPathSum(root: TreeNode?): Int {
    root ?: return 0
    return calculateMaxPathSum(root).second
}

fun calculateMaxPathSum(node: TreeNode?): Pair<Int, Int> {
    if (node == null) return 0 to Int.MIN_VALUE

    val (sumLeft, maxLeft) = calculateMaxPathSum(node.left)
    val (sumRight, maxRight) = calculateMaxPathSum(node.right)

    var sum = node.`val`
    sum = max(sum + sumLeft, sum + sumRight).coerceAtLeast(sum)

    var max = node.`val`
    max = max(max, max + sumLeft)
    max = max(max, max + sumRight)
    max = max(max, maxLeft)
    max = max(max, maxRight)

    return sum to max
}

fun findMaxPathSum(node: TreeNode?, map: MutableMap<TreeNode, Int>): Int {
    if (node == null) return Int.MIN_VALUE
    val sumLeft = map[node.left] ?: 0
    val sumRight = map[node.right] ?: 0

    var value = node.`val`
    value = max(value, value + sumLeft)
    value = max(value, value + sumRight)

    val left = findMaxPathSum(node.left, map)
    val right = findMaxPathSum(node.right, map)
    value = max(value, left)
    value = max(value, right)
    return value
}

fun calculateSum(node: TreeNode?, map: MutableMap<TreeNode, Int>): Int {
    if (node == null) return 0

    val left = calculateSum(node.left, map)
    val right = calculateSum(node.right, map)

    var value = node.`val`
    value = max(value + left, value + right).coerceAtLeast(value)
    map[node] = value
    return value
}

fun rob(root: TreeNode?): Int {
    fun dfs(node: TreeNode?): Pair<Int, Int> {
        if (node == null) return Pair(0, 0)

        val (leftRob, leftSkip) = dfs(node.left)
        val (rightRob, rightSkip) = dfs(node.right)

        val robThis = node.`val` + leftSkip + rightSkip
        val skipThis = max(leftRob, leftSkip) + max(rightRob, rightSkip)

        return Pair(robThis, skipThis)
    }

    val (robRoot, skipRoot) = dfs(root)
    return max(robRoot, skipRoot)
}

fun robGraphIndex(root: TreeNode?): Int {
    root ?: return 0
    // build graph
    val nodes = mutableListOf<Int>()
    val edges = mutableListOf<MutableList<Int>>()
    val parents = mutableListOf<Int>()

    val nodeToIndex = mutableMapOf<TreeNode, Int>()

    val stack = ArrayDeque<TreeNode>()
    stack.add(root)

    nodeToIndex[root] = 0
    nodes.add(root.`val`)
    edges.add(mutableListOf())
    parents.add(-1) // Root has no parent

    while (stack.isNotEmpty()) {
        val node = stack.removeLast()
        val index = nodeToIndex[node] ?: continue

        val childNodes = listOfNotNull(node.right, node.left)

        for (childNode in childNodes) {
            val childIndex = nodeToIndex.getOrPut(childNode) {
                val newIndex = nodes.size
                nodes.add(childNode.`val`)
                edges.add(mutableListOf())
                parents.add(index)
                newIndex
            }

            if (parents.size <= childIndex) {
                parents.add(index)
            } else if (parents[childIndex] == -1) {
                parents[childIndex] = index
            }

            edges[index].add(childIndex)
            stack.add(childNode)
        }
    }

    println(nodes.mapIndexed { index, i -> "$index ($i)" })
    //   println(edges.joinToString("\n") { list -> list.map { nodes[it].toString() }.toString()})
    val n = nodes.size
    val dp = IntArray(n)
    val visited = BooleanArray(n)
    val parent = IntArray(n) { -1 }

    fun visit(node: Int): Int {
        if (dp[node] > 0) return dp[node]

        visited[node] = true
        var grandChildrenValues = 0
        var childValues = 0
        for (childNode in edges[node]) {
            parent[childNode] = node
            val grandChildren = edges[childNode]
            childValues += visit(childNode)
            for (childrenNode in grandChildren) {
                //  if (visited[childrenNode]) continue
                parent[childrenNode] = childNode
                grandChildrenValues += visit(childrenNode)
            }
        }

        dp[node] = max(grandChildrenValues + nodes[node], childValues)
        println("Node: $node (${nodes[node]}): ${dp[node]}")
        return dp[node]
    }

    for (i in 0 until n) {
        if (dp[i] == 0) {
            visit(i)
        }
    }
    val rootValue = dp[0]
    val childrenValue = edges[0].sumOf { dp[it] }
    println(dp.toList())
    return max(rootValue, childrenValue)
}

fun superPow(a: Int, b: IntArray): Int {
    val mod = 1337
    val phi = 1140

    var exp = 0
    for (digit in b) {
        exp = (exp * 10 + digit) % phi
    }
    if (exp == 0) exp = phi

    var base = a % mod
    var result = 1
    var power = exp
    while (power > 0) {
        if (power and 1 == 1) result = (result * base) % mod
        base = (base * base) % mod
        power = power shr 1
    }
    return result
}

fun beautifulSubstrings(s: String, k: Int): Int {
    val n = s.length
    val vowels = setOf('u', 'e', 'o', 'a', 'i')
    val vowelsCount = IntArray(n + 1)
    val consonantsCount = IntArray(n + 1)

    for (i in 1..n) {
        if (s[i - 1] in vowels) {
            vowelsCount[i] = vowelsCount[i - 1] + 1
            consonantsCount[i] = consonantsCount[i - 1]
        } else {
            consonantsCount[i] = consonantsCount[i - 1] + 1
            vowelsCount[i] = vowelsCount[i - 1]
        }
    }

    var count = 0
    for (i in 1..(n - 1)) {
        for (j in (i + 1)..n) {
            val v = vowelsCount[j] - vowelsCount[i - 1]
            val c = consonantsCount[j] - consonantsCount[i - 1]
            if (v == c && v * c % k == 0) {
                count++
            }
        }
    }
    return count
}

fun countCharacters(words: Array<String>, chars: String): Int {
    val characters = IntArray('z'.code + 1)
    for (c in chars) characters[c.code]++

    var count = 0
    for (word in words) {
        val map = mutableMapOf<Int, Int>()
        var isValid = true
        for (c in word) {
            if (characters[c.code] == 0) {
                isValid = false
                break
            }
            map[c.code] = (map[c.code] ?: 0) + 1
        }
        if (!isValid) continue

        isValid = map.all { (key, wordCount) ->
            wordCount <= characters[key]
        }
        if (isValid) count += word.length
    }
    return count
}

fun calPoints(operations: Array<String>): Int {
    val scores = mutableListOf<Long>()
    for (operation in operations) {
        val num = operation.toLongOrNull()
        when {
            num != null -> {
                scores.add(num)
            }

            operation == "+" -> {
                val newScore = (scores.lastOrNull() ?: 0L) + (scores.getOrNull(scores.size - 2) ?: 0L)
                scores.add(newScore)
            }

            operation == "D" -> {
                val newScore = 2 * (scores.lastOrNull() ?: 0L)
                scores.add(newScore)
            }

            operation == "C" && scores.isNotEmpty() -> {
                scores.removeLast()
            }
        }
    }
    return scores.sum().toInt()
}


fun maximumGain(s: String, x: Int, y: Int): Int {
    val n = s.length
    if (n < 2) return 0

    val strings = mutableListOf<String>()
    var str = StringBuilder()
    for (i in 0 until s.length) {
        val c = s[i]
        if (i == s.length - 1 || (c != 'a' && c != 'b')) {
            if (i == s.length - 1 && (c == 'a' || c == 'b')) {
                str.append(c)
            }
            val subString = str.toString()
            if (subString.length > 1) {
                strings.add(subString)
            }
            str = StringBuilder()
            continue
        }
        str.append(c)
    }
    println(strings)

    var totalPoints = 0
    var char1 = 'a'
    var char2 = 'b'
    var point1 = x
    var point2 = y

    if (x < y) {
        char1 = 'b'
        char2 = 'a'
        point1 = y
        point2 = x
    }

    for (str in strings) {
        val firstStack = ArrayDeque<Char>()
        for (c in str) {
            if (firstStack.isNotEmpty()) {
                val top = firstStack.last()
                if (top == char1 && c == char2) {
                    firstStack.removeLast()
                    totalPoints += point1
                    continue
                }
            }
            firstStack.addLast(c)
        }

        val secondStack = ArrayDeque<Char>()
        for (c in firstStack) {
            if (secondStack.isNotEmpty()) {
                val top = secondStack.last()
                if (top == char2 && c == char1) {
                    secondStack.removeLast()
                    totalPoints += point2
                    continue
                }
            }
            secondStack.addLast(c)
        }
    }
    return totalPoints
}

fun numDecodings(s: String): Int {
    if (s.startsWith('0')) return 0
    val n = s.length
    if (n == 1) return 1
    val dp = IntArray(n)
    dp[0] = 1
    for (i in 1 until n) {
        if (s[i] != '0') dp[i] = dp[i - 1]
        if (s[i - 1] == '1' || s[i - 1] == '2') {
            val num = s.substring(i - 1, i + 1).toInt()
            if (num in 10..26) dp[i] += if (i >= 2) dp[i - 2] else 1
        }
        if (dp[i] == 0) return 0
    }
    return dp[n - 1]
}

fun countGoodStrings(low: Int, high: Int, zero: Int, one: Int): Int {
    val base = 1_000_000_007L
    val dp = LongArray(high + 1)
    dp[0] = 1L
    var count = 0L
    for (i in 1..high) {
        if (i >= zero) dp[i] = dp[i - zero] % base
        if (i >= one) dp[i] = (dp[i] + dp[i - one]) % base
        if (i >= low) count = (count + dp[i]) % base
    }
    return (count % base).toInt()
}

fun coinChange(coins: IntArray, amount: Int): Int {
    val dp = IntArray(amount + 1) { 100_000 }
    dp[0] = 0

    for (i in 0..amount) {
        for (coin in coins) {
            if (i >= coin) dp[i] = min(dp[i], dp[i - coin] + 1)
        }
    }

    return if (dp[amount] > amount) -1 else dp[amount]
}

fun mostPoints(questions: Array<IntArray>): Long {
    val n = questions.size
    val maxPointsSoFar = LongArray(n)
    maxPointsSoFar[0] = 0
    var totalMax = 0L
    for (i in 0 until n) {
        val (point, power) = questions[i]
        val pointSoFar = if (i > 0) maxPointsSoFar[i - 1] else 0
        val points = point.toLong() + pointSoFar
        maxPointsSoFar[i] = max(pointSoFar, maxPointsSoFar[i])
        val next = i + power
        if (next < n) {
            maxPointsSoFar[next] = max(maxPointsSoFar[next], points)
        }
        totalMax = max(totalMax, points)
    }

    return totalMax
}

fun findMaxForm(strs: Array<String>, m: Int, n: Int): Int {
    val size = strs.size
    val zeroList = mutableListOf<Int>()
    val oneList = mutableListOf<Int>()

    for (str in strs) {
        val zeroCount = str.count { it == '0' }
        zeroList.add(zeroCount)
        oneList.add(str.length - zeroCount)
    }

    val dp = Array(size + 1) { Array(m + 1) { IntArray(n + 1) } }

    for (i in 1..size) {
        val zero = zeroList[i - 1]
        val one = oneList[i - 1]
        for (j in 0..m) {
            for (k in 0..n) {
                var max = dp[i - 1][j][k]
                //  println("d ${i-1} ${j-zero} ${k-one} : ${dp[i - 1][j - zero][k - one]}")
                if (j >= zero && k >= one && (dp[i - 1][j - zero][k - one] + 1) > max) {
                    max = dp[i - 1][j - zero][k - one] + 1
                }
                dp[i][j][k] = max
            }
        }
    }
//    val res = mutableListOf<String>()
//    var i = size
//    var j = m
//    var k = n
//
//    while (i > 0) {
//        val zero = zeroList[i - 1]
//        val one = oneList[i - 1]
//
//        if (j >= zero && k >= one && dp[i][j][k] == dp[i - 1][j - zero][k - one] + 1) {
//            res.add(strs[i - 1])
//            j -= zero
//            k -= one
//            i--
//        } else {
//            i--
//        }
//    }
//    println(dp[size].print())
//    println(res)
    return dp[size][m][n]
}

fun combinationSum2(candidates: IntArray, target: Int): List<List<Int>> {
    val result = mutableListOf<List<Int>>()
    candidates.sort()

    fun dfs(start: Int, remain: Int, path: MutableList<Int>) {
        if (remain == 0) {
            result.add(path.toList())
            return
        }

        for (i in start until candidates.size) {
            if (i > start && candidates[i] == candidates[i - 1]) continue
            val num = candidates[i]
            if (num > remain) break
            path.add(num)
            dfs(i + 1, remain - num, path)
            path.removeAt(path.size - 1)
        }
    }

    dfs(0, target, mutableListOf())
    return result
}

fun combinationSumIterative(candidates: IntArray, target: Int): List<List<Int>> {
    val result = mutableListOf<List<Int>>()
    val stack = ArrayDeque<Pair<List<Int>, Int>>()

    stack.addLast(emptyList<Int>() to target)

    while (stack.isNotEmpty()) {
        val (comb, remain) = stack.removeLast()

        if (remain == 0) {
            result.add(comb)
            continue
        }

        val start = if (comb.isEmpty()) 0 else candidates.indexOf(comb.last())

        for (i in start until candidates.size) {
            val num = candidates[i]
            if (num <= remain) {
                stack.addLast(comb + num to remain - num)
            }
        }
    }

    return result
}

fun combinationSum(candidates: IntArray, target: Int): List<List<Int>> {

    val set = mutableSetOf<List<Int>>()
    fun dfs(candidates: IntArray, target: Int, list: List<Int>) {
        if (target == 0) {
            set.add(list.sorted())
            return
        }

        for (candidate in candidates) {
            if (target >= candidate) {
                dfs(candidates, target - candidate, list + candidate)
            }
        }
    }
    dfs(candidates, target, emptyList())

    return set.toList()
}


fun combinationSum4(nums: IntArray, target: Int): Int {
    val n = nums.size
    val dp = Array(target + 1) { 0 }
    dp[0] = 1

    for (i in 1..target) {
        for (num in nums) {
            if (i >= num) {
                dp[i] += dp[i - num]
            }
        }
    }

    return dp[target]
}

fun maximumUniqueSubarray(nums: IntArray): Int {
    var left = 0
    var right = 0
    val n = nums.size

    val lastSeen = IntArray(10001)
    val sumList = IntArray(n)
    var temporarySum = 0
    for ((i, num) in nums.withIndex()) {
        temporarySum += num
        sumList[i] = temporarySum
        lastSeen[num] = -1
    }

    var maxSum = 0
    while (right < n) {
        val num = nums[right]
        val lastIndex = lastSeen[num]

        if (lastIndex >= left) {
            val sum = sumList[right - 1] - sumList[left] + nums[left]
            maxSum = max(maxSum, sum)
            left = lastIndex + 1
        }
        lastSeen[num] = right
        right++
    }

    val lastSum = sumList[n - 1] - sumList[left] + nums[left]
    maxSum = max(maxSum, lastSum)

    return maxSum
}


fun change(amount: Int, coins: IntArray): Int {
    val n = coins.size
    val dp = Array(n + 1) { IntArray(amount + 1) }
    for (i in 0..n) dp[i][0] = 1

    for (i in 1..n) {
        val coin = coins[i - 1]
        for (j in 1..amount) {
            if (j >= coin) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - coin]
            } else {
                dp[i][j] = dp[i - 1][j]
            }
        }
    }
    return dp[n][amount]
}

fun makeFancyString(s: String): String {
    if (s.length < 3) return s
    var previousChar = s[0]
    var count = 1
    val result = StringBuilder().append(previousChar)

    for (i in 1 until s.length) {
        if (s[i] != previousChar) {
            count = 1
            previousChar = s[i]
            result.append(s[i])
        } else {
            count++
            if (count < 3) result.append(s[i])
        }
    }
    return result.toString()
}

fun missingNumber(nums: IntArray): Int {
    var result = 0

    for (i in 0 until nums.size) {
        result = result xor i xor nums[i]
    }
    return result
}

fun mySqrt(x: Int): Int {
    when {
        x == 0 -> return 0
        x < 4 -> return 1
        x < 9 -> return 2
        x < 16 -> return 3
    }
    val number = x.toLong()
    var start = 0
    var end = x / 3

    while (start < end) {
        val mid = start + (end - start) / 2
        val square = mid.toLong() * mid
        if (square == number) return mid
        if (square > number) {
            end = mid - 1
        }
        if (square < number) {
            start = mid + 1
        }
    }
    return if (start * start > number) start - 1 else start
}

fun lengthOfLastWord(s: String): Int {
    return s.trim().substringAfterLast(" ").length
}


fun numSquares(n: Int): Int {
    if (n <= 1) return 1

    val dp = IntArray(n + 1)
    dp[1] = 1

    for (i in 2..n) {
        val s = sqrt(i.toDouble()).toInt()
        dp[i] = n
        for (j in s downTo 1) {
            dp[i] = min(dp[i], dp[i - j * j] + 1)
        }
    }

    return dp[n]
}

fun numTrees(n: Int): Int {
    if (n == 0) return 0
    if (n == 1) return 1
    if (n == 2) return 2

    val dp = IntArray(n + 1)
    dp[0] = 1
    dp[1] = 1
    dp[2] = 2
    for (i in 3..n) {
        dp[i] = 0
        for (j in 0 until i) {
            dp[i] += dp[j] * dp[i - 1 - j]
        }
    }
    return dp[n]
}

fun maxProfit(prices: IntArray, fee: Int): Int {
    val n = prices.size
    if (n == 0) return 0

    val hold = IntArray(n)
    val cash = IntArray(n)

    hold[0] = -prices[0]

    for (i in 1 until n) {
        hold[i] = max(hold[i - 1], cash[i - 1] - prices[i])

        cash[i] = max(hold[i - 1] + prices[i] - fee, cash[i - 1])
    }

    return cash[n - 1]
}

fun maxProfit(numberTransactions: Int, prices: IntArray): Int {
    val n = prices.size
    if (n == 0) return 0
    val hold = Array(n) { IntArray(numberTransactions + 1) { 0 } }
    val cash = Array(n) { IntArray(numberTransactions + 1) { 0 } }

    hold[0][0] = 0
    for (k in 1..numberTransactions) {
        hold[0][k] = -prices[0]
    }

    for (i in 1 until n) {
        val price = prices[i]
        for (k in 1..numberTransactions) {
            hold[i][k] = max(
                hold[i - 1][k],
                cash[i - 1][k - 1] - price
            )
            cash[i][k] = max(
                hold[i - 1][k] + price,
                cash[i - 1][k]
            )
        }
    }
    //   println(hold.joinToString("\n") { it.toList().toString()})
    //  println()
    // println(cash.joinToString("\n") { it.toList().toString()})
    return cash[n - 1][numberTransactions]
}

fun maxProfit3(prices: IntArray): Int {
    val n = prices.size
    if (n == 0) return 0
    val numberTransactions = 2
    val hold = Array(n) { IntArray(numberTransactions + 1) { 0 } }
    val cash = Array(n) { IntArray(numberTransactions + 1) { 0 } }

    hold[0][0] = 0
    for (k in 1..numberTransactions) {
        hold[0][k] = -prices[0]
    }
    //   for(i in 0 until n) hold[i][0] = Int.MIN_VALUE

    for (i in 1 until n) {
        val price = prices[i]
        for (k in 1..numberTransactions) {
            hold[i][k] = max(
                hold[i - 1][k],
                cash[i - 1][k - 1] - price
            )
            cash[i][k] = max(
                hold[i - 1][k] + price,
                cash[i - 1][k]
            )
        }
    }
    //   println(hold.joinToString("\n") { it.toList().toString()})
    //  println()
    // println(cash.joinToString("\n") { it.toList().toString()})
    return cash[n - 1][numberTransactions]
}

fun maxProfit23(prices: IntArray): Int {
    val n = prices.size
    if (n == 0) return 0

    var total = 0
    for (i in 1 until n) {
        if (prices[i] > prices[i - 1]) {
            total += (prices[i] - prices[i - 1])
        }
    }

    return total
}

fun maxProfit2(prices: IntArray): Int {
    val n = prices.size
    if (n == 0) return 0

    val hold = IntArray(n)
    val cash = IntArray(n)

    hold[0] = -prices[0]

    for (i in 1 until n) {
        hold[i] = max(hold[i - 1], cash[i - 1] - prices[i])

        cash[i] = max(hold[i - 1] + prices[i], cash[i - 1])
    }

    return cash[n - 1]
}

fun maxProfit309(prices: IntArray): Int {
    val n = prices.size
    if (n == 0) return 0

    val hold = IntArray(n)
    val sold = IntArray(n)
    val rest = IntArray(n)

    hold[0] = -prices[0]
    sold[0] = 0
    rest[0] = 0

    for (i in 1 until n) {
        hold[i] = max(hold[i - 1], rest[i - 1] - prices[i])
        sold[i] = hold[i - 1] + prices[i]
        rest[i] = max(rest[i - 1], sold[i - 1])
    }

    return max(sold[n - 1], rest[n - 1])
}

fun wordBreak(s: String, wordDict: List<String>): Boolean {
    if (s.isEmpty() && wordDict.isEmpty()) return true
    if (s.isEmpty() || wordDict.isEmpty()) return false
    val m = s.length
    val n = wordDict.size
    val dp = BooleanArray(m + 1)

    dp[0] = true

    for (i in 1..m) {
        dp[i] = wordDict.any { word ->
            i > word.length && dp[i - word.length - 1] && s.takeLast(word.length) == word
        }
    }
    println(dp.toList())
    return dp[m]
}

fun longestPalindrome(s: String): String {
    if (s.isEmpty()) return ""
    if (s.length == 1) return s
    val n = s.length
    val dp = Array(n) { IntArray(n) { -1 } }
    for (i in 0 until n) dp[i][i] = 1
    for (i in 0 until n - 1) dp[i][i + 1] = if (s[i] == s[i + 1]) 1 else 0


    fun check(s: String, i: Int, j: Int): Int {
        if (i > j || i !in s.indices || j !in s.indices) return -1
        if (dp[i][j] != -1) return dp[i][j]
        if (i == j) return 1
        return if (s[i] != s[j]) 0 else check(s, i + 1, j - 1)
    }

    for (j in n - 1 downTo 0) {
        for (i in 0 until j) {
            if (dp[i][j] == -1) dp[i][j] = check(s, i, j)
        }
    }

    var max = 0
    var start = 0
    var end = 0
    for (i in 0 until n) {
        for (j in i until n) {
            if (dp[i][j] != 1) continue
            if (max < j - i + 1) {
                max = j - i + 1
                start = i
                end = j
            }
        }
    }
    // println(dp.joinToString("\n") { it.toList().toString() })
    return s.substring(start, end + 1)
}


fun findLUSlength(strs: Array<String>): Int {
    if (strs.isEmpty()) return -1
    val set = strs.toSet()
    if (set.size == 1) return -1
    var max = -1
    for (i in 0 until strs.size) {
        var hasDuplicate = false
        for (j in 0 until strs.size) {
            if (i == j) continue
            if (isSubsequence(strs[i], strs[j])) {
                hasDuplicate = true
                break
            }
        }
        if (!hasDuplicate) {
            max = max(max, strs[i].length)
        }
    }
    return max
}

fun isSubsequence(s: String, t: String): Boolean {
    var i = 0
    var j = 0

    while (i < s.length && j < t.length) {
        if (s[i] == t[j]) {
            i++
        }
        j++
    }

    return i == s.length
}

fun findMinimumOperations(s1: String, s2: String, s3: String): Int {
    if (s1 == s2 && s2 == s3) return 0

    val length = min(s1.length, min(s2.length, s3.length))
    if (length == 0) return -1

    var prefixCount = 0
    while (prefixCount < length) {
        val prefix = s1.take(prefixCount + 1)
        if (s1.startsWith(prefix) && s2.startsWith(prefix) && s3.startsWith(prefix)) {
            prefixCount++
        } else break
    }
    if (prefixCount == 0) {
        return -1
    }
    val sumOfLengths = s1.length + s2.length + s3.length
    return sumOfLengths - 3 * prefixCount
}

fun minDistance(word1: String, word2: String): Int {
    val m = word1.length
    val n = word2.length

    if (m == 0 || n == 0) return m + n

    val dp = Array(m + 1) { IntArray(n + 1) }
    for (i in 0..m) dp[i][0] = i
    for (j in 0..n) dp[0][j] = j

    for (i in 1..m) {
        for (j in 1..n) {
            dp[i][j] = if (word1[i - 1] == word2[j - 1]) {
                dp[i - 1][j - 1]
            } else {
                min(min(dp[i - 1][j], dp[i][j - 1]), dp[i - 1][j - 1]) + 1
            }
        }
    }

    return dp[m][n]
}

fun minimumDeleteSum(s1: String, s2: String): Int {
    if (s1.isEmpty() || s2.isEmpty()) return 0
    val m = s1.length
    val n = s2.length

    val dp = Array(m + 1) { IntArray(n + 1) }
    val ascii = Array(m + 1) { IntArray(n + 1) }

    for (i in 0..m) {
        ascii[i][0] = 0
        dp[i][0] = 0
    }
    for (j in 0..n) {
        dp[0][j] = 0
        ascii[0][j] = 0
    }

    for (i in 1..m) {
        for (j in 1..n) {
            var maxAscii = 0
            dp[i][j] = if (s1[i - 1] == s2[j - 1]) {
                dp[i - 1][j - 1] + 1
            } else {
                max(dp[i - 1][j], dp[i][j - 1])
            }

            if (s1[i - 1] == s2[j - 1]) {
                maxAscii = max(maxAscii, ascii[i - 1][j - 1] + s1[i - 1].code)
            }
            if (s1[i - 1] != s2[j - 1] && dp[i][j] == dp[i - 1][j]) {
                maxAscii = max(maxAscii, ascii[i - 1][j])
            }
            if (s1[i - 1] != s2[j - 1] && dp[i][j] == dp[i][j - 1]) {
                maxAscii = max(maxAscii, ascii[i][j - 1])
            }

            ascii[i][j] = maxAscii
        }
    }


    var sumAscii = 0
    for (i in 0 until m) {
        sumAscii += s1[i].code
    }
    for (i in 0 until n) {
        sumAscii += s2[i].code
    }
    return sumAscii - 2 * ascii[m][n]
}

fun findMaxAsciiString(dp: Array<IntArray>, s1: String, s2: String, m: Int, n: Int): Int {
    if (m == 0 || n == 0 || dp[m][n] == 0) return 0
    var max = 0
    if (dp[m][n] == dp[m - 1][n - 1] + 1 && s1[m - 1] == s2[n - 1]) {
        max = max(max, findMaxAsciiString(dp, s1, s2, m - 1, n - 1) + s1[m - 1].code)
        return max
    }
    if (dp[m][n] == dp[m - 1][n] && s1[m - 1] != s2[n - 1]) {
        max = max(max, findMaxAsciiString(dp, s1, s2, m - 1, n))
    }
    if (dp[m][n] == dp[m][n - 1] && s1[m - 1] != s2[n - 1]) {
        max = max(max, findMaxAsciiString(dp, s1, s2, m, n - 1))
    }
    return max
}

fun minDistance2(word1: String, word2: String): Int {
    if (word1.isEmpty() || word2.isEmpty()) return 0
    val m = word1.length
    val n = word2.length

    val dp = Array(m + 1) { IntArray(n + 1) }
    for (i in 0..m) dp[i][0] = 0
    for (j in 0..n) dp[0][j] = 0

    for (i in 1..m) {
        for (j in 1..n) {
            dp[i][j] = if (word1[i - 1] == word2[j - 1]) {
                dp[i - 1][j - 1] + 1
            } else {
                max(dp[i - 1][j], dp[i][j - 1])
            }
        }
    }

    return word1.length - dp[m][n] + word2.length - dp[m][n]
}

fun longestCommonSubsequence(text1: String, text2: String): Int {
    if (text1.isEmpty() || text2.isEmpty()) return 0
    val m = text1.length
    val n = text2.length

    val dp = Array(m + 1) { IntArray(n + 1) }
    for (i in 0..m) dp[i][0] = 0
    for (j in 0..n) dp[0][j] = 0

    for (i in 1..m) {
        for (j in 1..n) {
            dp[i][j] = if (text1[i - 1] == text2[j - 1]) {
                dp[i - 1][j - 1] + 1
            } else {
                max(dp[i - 1][j], dp[i][j - 1])
            }
        }
    }

    return dp[m][n]
}

fun maximalSquare(matrix: Array<CharArray>): Int {
    val m = matrix.size
    val n = matrix[0].size

    val dp = Array(m) { IntArray(n) }
    dp[0][0] = if (matrix[0][0] == '1') 1 else 0
    for (i in 1 until m) dp[i][0] = if (matrix[i][0] == '1') 1 else 0
    for (j in 1 until n) dp[0][j] = if (matrix[0][j] == '1') 1 else 0

    for (i in 1 until m) {
        for (j in 1 until n) {
            dp[i][j] = if (matrix[i][j] == '1') {
                min(dp[i - 1][j - 1], min(dp[i][j - 1], dp[i - 1][j])) + 1
            } else 0
        }
    }

    var maxValue = -1
    for (i in 0 until m) {
        for (j in 0 until n) {
            maxValue = max(maxValue, dp[i][j])
        }
    }
    return maxValue * maxValue
}

fun minFallingPathSum(matrix: Array<IntArray>): Int {
    if (matrix.isEmpty()) return 0
    val m = matrix.size
    val n = m
    if (n == 0) return 0

    val dp = Array(m) {
        IntArray(n + 1) { 10005 }
    }
    dp[0][0] = matrix[0][0]

    for (i in 1 until m) dp[i][0] = dp[i - 1][0] + matrix[i][0]
    for (j in 1 until n) dp[0][j] = matrix[0][j]

    for (i in 1 until m) {
        for (j in 1 until n) {
            dp[i][j] = matrix[i][j] + min(min(dp[i - 1][j], dp[i - 1][j - 1]), dp[i - 1][j + 1])
        }
    }
    println(dp[m - 1].toList())
    return dp[m - 1].min()
}

fun minimumTotal(triangle: List<List<Int>>): Int {
    if (triangle.isEmpty()) return 0
    val m = triangle.size
    val n = triangle[m - 1].size
    if (n == 0) return 0

    val dp = Array(m) {
        IntArray(n) { 10005 }
    }
    dp[0][0] = triangle[0][0]
    for (i in 1 until m) dp[i][0] = dp[i - 1][0] + triangle[i][0]
    for (j in 1 until n) dp[0][j] = triangle[0][0]

    for (i in 1 until m) {
        for (j in 1 until triangle[i].size) {
            dp[i][j] = triangle[i][j] + min(dp[i - 1][j], dp[i - 1][j - 1])
        }
    }
    return dp[m - 1].min()
}

fun uniquePathsWithObstacles(obstacleGrid: Array<IntArray>): Int {
    val m = obstacleGrid.size
    val n = obstacleGrid[0].size
    if (m == 0 || n == 0) return 0

    val dp = Array(m) {
        IntArray(n)
    }
    dp[0][0] = if (obstacleGrid[0][0] == 1) 0 else 1
    for (i in 1 until m) dp[i][0] = if (obstacleGrid[i][0] == 1) 0 else dp[i - 1][0]
    for (j in 1 until n) dp[0][j] = if (obstacleGrid[0][j] == 1) 0 else dp[0][j - 1]

    for (i in 1 until m) {
        for (j in 1 until n) {
            dp[i][j] = if (obstacleGrid[i][j] == 1) 0 else dp[i - 1][j] + dp[i][j - 1]
        }
    }
    return dp[m - 1][n - 1]
}

fun minPathSum(grid: Array<IntArray>): Int {
    val m = grid.size
    val n = grid[0].size

    val dp = Array(m) {
        IntArray(n)
    }
    dp[0][0] = grid[0][0]
    for (i in 1 until m) dp[i][0] = grid[i][0] + dp[i - 1][0]
    for (j in 1 until n) dp[0][j] = grid[0][j] + dp[0][j - 1]

    for (i in 1 until m) {
        for (j in 1 until n) {
            dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j]
        }
    }
    return dp[m - 1][n - 1]
}

fun uniquePaths(m: Int, n: Int): Int {
    val dp = Array(m) {
        IntArray(n)
    }

    for (i in 0 until m) dp[i][0] = i + 1
    for (j in 0 until m) dp[0][j] = j + 1

    for (i in 1 until m) {
        for (j in 1 until n) {
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1] + 1
        }
    }
    return dp[m - 1][n - 1]
}

fun deleteAndEarn(nums: IntArray): Int {
    val maxNumber = nums.maxOrNull() ?: return 0
    val sums = LongArray(maxNumber + 1) { 0 }
    for (num in nums) {
        sums[num] += num
    }

    val dp = LongArray(maxNumber + 1)
    dp[0] = sums[0]
    dp[1] = max(sums[0], sums[1])
    for (i in 2..maxNumber) {
        val value = sums[i]
        dp[i] = max(dp[i - 1], dp[i - 2] + value)
    }
    return dp[maxNumber].toInt()
}

fun rob(nums: IntArray): Int {
    val n = nums.size
    when (n) {
        0 -> return 0
        1 -> return nums[0]
        2 -> return max(nums[0], nums[1])
    }

    val dp = IntArray(n + 1) { 0 }
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])

    for (i in 2 until n) {
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])
    }
    return dp.maxOrNull() ?: 0
}

fun minCostClimbingStairs(cost: IntArray): Int {
    val n = cost.size
    val dp = IntArray(n + 1)
    if (n == 0) return 0
    if (n == 1) return cost[0]
    if (n == 2) return min(cost[0], cost[1])

    dp[0] = cost[0]
    dp[1] = cost[1]

    for (i in 2..n) {
        dp[i] = min(dp[i - 1], dp[i - 2]) + (cost.getOrNull(i) ?: 0)
    }
    return dp[n]
}

fun tribonacci(n: Int): Int {
    if (n == 0) return 0
    if (n == 1) return 1
    if (n == 2) return 1

    val dp = IntArray(n + 1)
    dp[0] = 0
    dp[1] = 1
    dp[2] = 1
    for (i in 3..n) {
        dp[i] = dp[i - 1] + dp[i - 2] + dp[i - 3]
    }
    return dp[n]
}

fun fib(n: Int): Int {
    if (n == 0) return 0
    if (n == 1) return 1
    val dp = IntArray(n + 1)
    dp[0] = 0
    dp[1] = 1

    for (i in 2..n) {
        dp[i] = dp[i - 1] + dp[i - 2]
    }

    return dp[n]
}

fun climbStairs(n: Int): Int {

    val dp = IntArray(n + 1)
    dp[0] = 1
    dp[1] = 1

    for (i in 2..n) {
        dp[i] = dp[i - 1] + dp[i - 2]
    }

    return dp[n]
}

fun canPartition(nums: IntArray): Boolean {
    val sum = nums.sum()
    if (sum % 2 != 0) return false
    val target = sum / 2
    val dp = BooleanArray(sum + 1) { false }
    dp[0] = true

    for (i in nums) {
        for (j in target downTo i) {
            dp[j] = dp[j] || dp[j - i]
        }
    }

    return dp[target]
}

fun minAbsDifference(nums: IntArray, goal: Int): Int {
    val n = nums.size
    val left = nums.slice(0 until n / 2)
    val right = nums.slice(n / 2 until n)

    val leftSums = subsetSums(left)
    val rightSums = subsetSums(right).sorted()

    var res = Int.MAX_VALUE
    for (l in leftSums) {
        val remain = goal - l
        val idx = binarySearchClosest(rightSums, remain)
        val r = rightSums[idx]
        res = minOf(res, abs(l + r - goal))
        if (res == 0) return 0
    }

    return res
}

fun subsetSums(arr: List<Int>): List<Int> {
    val res = mutableListOf(0)
    for (x in arr) {
        val newSums = res.map { it + x }
        res.addAll(newSums)
    }
    return res
}

fun binarySearchClosest(list: List<Int>, value: Int): Int {
    var low = 0
    var high = list.size - 1
    var best = 0
    while (low <= high) {
        val mid = (low + high) / 2
        if (list[mid] == value) return mid
        if (abs(list[mid] - value) < abs(list[best] - value)) best = mid
        if (list[mid] < value) low = mid + 1 else high = mid - 1
    }
    return best
}

fun lastStoneWeightII(stones: IntArray): Int {
    val sum = stones.sum()
    val set = mutableSetOf<Int>()

    for (i in stones) {
        set.addAll(set.map { it + i } + i)
    }


    return set.minOfOrNull { abs(sum - 2 * it) } ?: 0
}

fun findMaxFish(grid: Array<IntArray>): Int {
    val m = grid.size
    val n = grid[0].size


    var value = 0

    for (i in 0 until m) {
        for (j in 0 until n) {
            if (grid[i][j] > 0) {
                value = max(value, dfsFish(grid, i, j, m, n))
            }
        }
    }
    return value
}

private fun dfsFish(grid: Array<IntArray>, x: Int, y: Int, m: Int, n: Int): Int {
    if (x !in 0 until m || y !in 0 until n || grid[x][y] == 0) return 0

    val value = grid[x][y]
    grid[x][y] = 0
    return value + dfsFish(grid, x + 1, y, m, n) +
            dfsFish(grid, x - 1, y, m, n) +
            dfsFish(grid, x, y + 1, m, n) +
            dfsFish(grid, x, y - 1, m, n)
}

fun countSubIslands2(grid1: Array<IntArray>, grid2: Array<IntArray>): Int {
    val m = grid2.size
    val n = grid2[0].size

    for (i in 0 until m) {
        for (j in 0 until n) {
            if (grid2[i][j] == 1 && grid1[i][j] == 0) {
                dfs(grid2, i, j, m, n) // xóa đảo này
            }
        }
    }

    var count = 0

    for (i in 0 until m) {
        for (j in 0 until n) {
            if (grid2[i][j] == 1) {
                count++
                dfs(grid2, i, j, m, n)
            }
        }
    }
    return count
}

private fun dfs(grid: Array<IntArray>, x: Int, y: Int, m: Int, n: Int) {
    if (x !in 0 until m || y !in 0 until n || grid[x][y] == 0) return

    grid[x][y] = 0

    dfs(grid, x + 1, y, m, n)
    dfs(grid, x - 1, y, m, n)
    dfs(grid, x, y + 1, m, n)
    dfs(grid, x, y - 1, m, n)
}

fun countSubIslands(grid1: Array<IntArray>, grid2: Array<IntArray>): Int {
    val m = grid2.size
    val n = grid2.firstOrNull()?.size ?: 0

    var count = 0
    for (i in 0 until m) {
        for (j in 0 until n) {
            if (grid2[i][j] == 1 && grid1[i][j] == 1) {
                val isSubIsland = fillIslandArea(grid1, grid2, i, j, m, n, 2)
                if (isSubIsland == 1) {
                    count++
                }
            }
        }
    }
    return count
}

private fun fillIslandArea(
    grid1: Array<IntArray>,
    grid2: Array<IntArray>,
    x: Int,
    y: Int,
    m: Int,
    n: Int,
    color: Int
): Int {
    if (x !in 0 until m || y !in 0 until n || grid2[x][y] == 0) {
        return 1
    }
    if (grid2[x][y] == color) return 1

    grid2[x][y] = color
    var value = if (grid1[x][y] == 0) 0 else 1
    listOf(
        fillIslandArea(grid1, grid2, x + 1, y, m, n, color),
        fillIslandArea(grid1, grid2, x - 1, y, m, n, color),
        fillIslandArea(grid1, grid2, x, y + 1, m, n, color),
        fillIslandArea(grid1, grid2, x, y - 1, m, n, color),
    ).forEach {
        value *= it
    }

    return value
}

fun islandPerimeter(grid: Array<IntArray>): Int {
    val m = grid.size
    val n = grid.firstOrNull()?.size ?: 0

    var sX = -1
    var sY = -1
    for (x in grid.indices) {
        for (y in grid[x].indices) {
            if (grid[x][y] == 1) {
                sX = x
                sY = y
                break
            }
        }
        if (sX >= 0 && sY >= 0) break
    }
    return calculatePerimeter(grid, sX, sY, m, n, 2)
}

private fun calculatePerimeter(image: Array<IntArray>, x: Int, y: Int, m: Int, n: Int, color: Int): Int {
    if (x !in 0 until m || y !in 0 until n || image[x][y] == 0) {
        return 1
    }
    if (image[x][y] == color) return 0

    image[x][y] = color
    var sum = 0
    sum += calculatePerimeter(image, x + 1, y, m, n, color)
    sum += calculatePerimeter(image, x - 1, y, m, n, color)
    sum += calculatePerimeter(image, x, y + 1, m, n, color)
    sum += calculatePerimeter(image, x, y - 1, m, n, color)
    return sum
}

val xDir = intArrayOf(-1, 0, 0, 1)
val yDir = intArrayOf(0, -1, 1, 0)
fun floodFill(image: Array<IntArray>, sr: Int, sc: Int, color: Int): Array<IntArray> {
    if (image[sr][sc] == color) return image
    val m = image.size
    val n = image.firstOrNull()?.size ?: 0
    fill(image, sr, sc, m, n, color)
    return image
}

private fun fill(image: Array<IntArray>, x: Int, y: Int, m: Int, n: Int, color: Int) {
    val previousColor = image[x][y]
    image[x][y] = color

    for (i in 0 until 4) {
        val newX = x + xDir[i]
        val newY = y + yDir[i]
        if (newX !in 0 until m || newY !in 0 until n || image[newX][newY] != previousColor) continue
        fill(image, newX, newY, m, n, color)
    }

}

fun canPlaceFlowers(flowerbed: IntArray, n: Int): Boolean {
    var start = -1
    var count = 0
    for (i in 0 until flowerbed.size) {
        when {
            flowerbed[i] == 0 && i == flowerbed.size - 1 && start >= 0 -> {
                val newFlowers = (flowerbed.size - start - 1) / 2
                count += newFlowers
            }

            flowerbed[i] == 0 && i == flowerbed.size - 1 -> {
                count += (flowerbed.size - 1) / 2 + 1
            }

            flowerbed[i] != 1 -> continue
            else -> {
                val newFlowers = if (start == -1) {
                    i / 2
                } else {
                    val zeroCount = i - start - 1
                    (zeroCount / 2f).roundToInt() - 1
                }
                if (newFlowers > 0) count += newFlowers
            }
        }
        start = i
    }
    return count >= n
}

fun majorityElement(nums: IntArray): List<Int> {
    return nums.groupBy { it }.filter { it.value.size > (nums.size / 3) }.map { it.key }
}


fun containsNearbyDuplicate(nums: IntArray, k: Int): Boolean {

    val delta = k.coerceAtMost(nums.size)

    val set = mutableSetOf<Int>()
    val window = mutableListOf<Int>()
    for (i in 0 until delta) {
        if (nums[i] in set) return true
        set.add(nums[i])
        window.add(nums[i])
    }

    for (i in delta until nums.size) {
        if (nums[i] in set) return true
        set.add(nums[i])
        window.add(nums[i])
        val x = window.removeAt(0)
        set.remove(x)
    }
    return false
}

fun lastStoneWeight(stones: IntArray): Int {
    var list = stones.sorted()

    while (list.size > 1) {
        list = list.sorted()
        val (first, second) = list.takeLast(2)
        val newStones = listOfNotNull(abs(first - second).takeIf { it > 0 })
        list = (list.dropLast(2) + newStones).sorted()
    }
    return list.firstOrNull() ?: 0
}

fun heightChecker(heights: IntArray): Int {
    val sortedArray = heights.sorted()
    var count = 0
    for (i in 0 until sortedArray.size) {
        if (heights[i] != sortedArray[i]) count++
    }
    return count
}

fun isPossibleToSplit(nums: IntArray): Boolean {
    return nums.groupBy { it }.all { it.value.size <= 2 }
}

fun searchRange(nums: IntArray, target: Int): IntArray {
    if (nums.isEmpty()) return intArrayOf(-1, -1)
    if (nums[0] == nums[nums.size - 1]) {
        if (nums[0] == target) return intArrayOf(0, nums.size - 1)
        return intArrayOf(-1, -1)
    }
    val index = nums.binarySearch(target)
    if (index < 0) return intArrayOf(-1, -1)
    var start = index
    var end = index
    while (start >= 0 && nums[start] == nums[index]) start--
    while (end < nums.size && nums[end] == nums[index]) end++

    return intArrayOf(
        (start + 1).coerceAtLeast(0),
        (end - 1).coerceAtMost(nums.size - 1)
    )
}


fun duplicateZeros(arr: IntArray): Unit {
    var index = 0
    val size = arr.count { it == 0 } + arr.size
    val newArray = IntArray(size)
    for (i in 0 until arr.size) {
        newArray[index++] = arr[i]
        if (arr[i] == 0) {
            newArray[index++] = 0
        }
    }
    for (i in 0 until size) arr[i] = newArray[i]
    println(arr.toList())
}

fun getRow(rowIndex: Int): List<Int> {
    val row = MutableList(rowIndex + 1) { 1 }
    for (k in 1 until rowIndex) {
        row[k] = (row[k - 1].toLong() * (rowIndex - k + 1) / k).toInt()
    }
    return row
}

fun generate(numRows: Int): List<List<Int>> {
    if (numRows == 0) return emptyList()
    if (numRows == 1) return listOf(listOf(1))
    if (numRows == 2) return listOf(listOf(1), listOf(1, 1))

    val rows = mutableListOf(listOf(1))

    var previousRow = listOf(1, 1)
    for (row in 1 until numRows) {
        val newRow = List(row) {
            previousRow[it] + (previousRow.getOrNull(it - 1) ?: 0)
        } + 1
        rows.add(newRow)
        previousRow = newRow
    }
    return rows
}

fun plusOne(digits: IntArray): IntArray {
    val array = BigInteger(digits.joinToString(""))
        .plus(BigInteger.ONE)
        .toString()
    return IntArray(array.length) { array[it].digitToInt() }
}

fun searchInsert(nums: IntArray, target: Int): Int {
    var left = 0
    var right = nums.size - 1

    while (left <= right) {
        val mid = left + (right - left) / 2
        when {
            nums[mid] == target -> return mid
            nums[mid] < target -> left = mid + 1
            else -> right = mid - 1
        }
    }

    return left
}

fun removeElement(nums: IntArray, `val`: Int): Int {
    var k = 0

    for (i in nums.indices) {
        if (nums[i] != `val`) {
            nums[k] = nums[i]
            k++
        }
    }
    println("$k, ${nums.take(k).toList()}")
    return k
}

fun removeDuplicates(nums: IntArray): Int {
    val distinstArray = nums.distinct()
    val length = distinstArray.size
    for (i in 0 until length) nums[i] = distinstArray[i]
    return length
}

fun longestCommonPrefix(strs: Array<String>): String {
    if (strs.isEmpty()) return ""
    if (strs.size == 1) return strs[0]

    val first = strs[0]
    if (first.isEmpty()) return ""

    for (i in first.length - 1 downTo 0) {
        val prefix = first.take(i)
        val isCommonPrefix = strs.all { it.startsWith(prefix) }
        if (isCommonPrefix) return prefix
    }
    return ""
}

fun findWordsContaining(words: Array<String>, x: Char): List<Int> {
    return words.mapIndexedNotNull { index, word ->
        if (word.contains(x)) index else null
    }
}

fun buildArray(nums: IntArray): IntArray {
    return IntArray(nums.size) { nums[nums[it]] }
}

fun smallerNumbersThanCurrent(nums: IntArray): IntArray {
    return IntArray(nums.size) { index ->
        var count = 0
        for (i in nums.indices) {
            if (i != index && nums[i] < nums[index]) count++
        }
        count
    }
}

fun minMovesToSeat(seats: IntArray, students: IntArray): Int {
    return abs(seats.sum() - students.sum())
}

fun countPairs(nums: List<Int>, target: Int): Int {
    val numbers = nums.sorted()
    var count = 0
    for (i in 0 until numbers.size - 1) {
        for (j in i + 1 until numbers.size) {
            val sum = numbers[i] + numbers[j]
            if (sum >= target) break
            count++
        }
    }
    return count
}

fun transformArray(nums: IntArray): IntArray {
    if (nums.isEmpty()) {
        return nums
    }
    var evenCount = 0
    var oddCount = 0
    for (num in nums) {
        if (num % 2 != 0) oddCount++ else evenCount++
    }

    var index = 0
    while (index < evenCount) nums[index++] = 0
    while (index < nums.size) nums[index++] = 1
    return nums
}

fun sortColors(nums: IntArray): Unit {
    if (nums.isEmpty()) {
        println("[]")
        return
    }
    val colors = IntArray(3) { 0 }
    for (color in nums) {
        colors[color]++
    }
    val first = IntArray(colors[0]) { 0 }
    val second = IntArray(colors[1]) { 1 }
    val third = IntArray(colors[2]) { 2 }
    val result = first + second + third
    for (i in result.indices) nums[i] = result[i]
    println(nums.joinToString(prefix = "[", postfix = "]", separator = ","))
}

fun merge(intervals: Array<IntArray>): Array<IntArray> {
    return mergeList(intervals)
}

fun mergeList(intervals: Array<IntArray>): Array<IntArray> {
    val intervalList = mutableListOf<IntArray>()
    for (interval in intervals) {
        var hasIntersect = false
        val intervalRange = interval[0]..interval[1]
        for (i in intervalList.indices) {
            val first = intervalList[i][0]
            val second = intervalList[i][1]
            val range = first..second
            val isIntersect = interval[0] in range || interval[1] in range
                    || first in intervalRange || second in intervalRange

            if (isIntersect) {
                hasIntersect = true
                val start = min(intervalList[i][0], interval[0])
                val end = max(intervalList[i][1], interval[1])
                intervalList[i] = intArrayOf(start, end)
            }
        }
        if (!hasIntersect) {
            intervalList.add(interval)
        }
    }

    val output = intervalList.distinctBy { it[0] to it[1] }.toTypedArray()
    val outputString = output.joinToString(prefix = "[", postfix = "]") { it.joinToString(prefix = "[", postfix = "]") }
    val inputString =
        intervals.joinToString(prefix = "[", postfix = "]") { it.joinToString(prefix = "[", postfix = "]") }
    return if (outputString == inputString) output else merge(output)
}

fun groupAnagrams(strs: Array<String>): List<List<String>> {
    return strs.groupBy { it.toList().sorted() }.values.toList()
}

fun search(nums: IntArray, target: Int): Int {
    if (nums.isEmpty()) return -1
    if (nums.size < 20) return nums.indexOf(target)

    var left = 0
    var right = nums.size - 1

    while (left <= right) {
        val mid = left + (right - left) / 2
        val midValue = nums[mid]
        if (midValue == target) return mid

        when {
            nums[left] < midValue -> {
                if (target <= midValue && target >= nums[left]) {
                    return binarySearchInRange(nums, target, left, mid)
                }
                left = mid + 1
            }

            nums[right] > midValue -> {
                if (target >= midValue && target <= nums[right]) {
                    return binarySearchInRange(nums, target, mid, right)
                }
                right = mid - 1
            }

            else -> {
                while (left < nums.size && nums[left] == midValue) left++
                while (right >= 0 && nums[right] == midValue) right--
            }
        }
    }

    return -1
}

fun binarySearchInRange(
    arr: IntArray,
    target: Int,
    start: Int,
    end: Int,
): Int {
    var left = start
    var right = end

    while (left <= right) {
        val mid = left + (right - left) / 2
        when {
            arr[mid] == target -> return mid
            arr[mid] < target -> left = mid + 1
            else -> right = mid - 1
        }
    }

    return -1
}

fun toListNode(list: List<Int>): ListNode? {
    if (list.isEmpty()) return null

    val dummy = ListNode(0)
    var current = dummy

    for (value in list) {
        current.next = ListNode(value)
        current = current.next!!
    }

    return dummy.next
}
