package local

import java.util.*
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min


fun countEven(num: Int): Int {
    if (num < 2) return 0
    var cnt = 0
    for (i in 2..num) {
        val digitSum = num.toString().sumOf { it.digitToInt() }
        if (digitSum % 2 == 0) cnt++
    }
    return cnt
}

fun maxTotalFruits(fruits: Array<IntArray>, startPos: Int, k: Int): Int {
    fun getRangeSum(treeMap: TreeMap<Int, Long>, left: Int, right: Int): Long {
        val sumRight = treeMap.floorEntry(right)?.value ?: 0L
        val sumLeft = treeMap.floorEntry(left - 1)?.value ?: 0L
        return sumRight - sumLeft
    }

    val n = fruits.size
    val treeMap = TreeMap<Int, Long>()
    var preSum = 0L
    for (i in 0 until n) {
        val (pos, amount) = fruits[i]
        val sum = preSum + amount
        treeMap[pos] = sum
        preSum = sum
    }

    var maxAmount = maxOf(
        getRangeSum(treeMap, startPos - k, startPos),
        getRangeSum(treeMap, startPos, startPos + k)
    )

    for (x in 0..(k / 2)) {
        maxAmount = maxOf(
            maxAmount,
            getRangeSum(treeMap, startPos - k + 2 * x, startPos + x),
            getRangeSum(treeMap, startPos - x, startPos + k - 2 * x),
        )
    }
    return maxAmount.toInt()
}

fun waysToBuyPensPencils(total: Int, cost1: Int, cost2: Int): Long {
    var count = 0L
    var x = 0
    while (cost1 * x <= total) {
        val remaining = total - cost1 * x
        val maxY = remaining / cost2
        count += (maxY + 1)
        x++
    }
    return count
}

fun sumOfThree(num: Long): LongArray {
    if (num % 3 != 0L) return longArrayOf()
    val x = num / 3L
    return longArrayOf(x - 1, x, x + 1)
}

fun maxConsecutive(bottom: Int, top: Int, special: IntArray): Int {
    special.sort()
    val n = special.size
    var previousFloor = bottom.toLong()
    var maxLength = 0L
    for (i in 0 until n) {
        val floor = special[i].toLong()
        maxLength = max(maxLength, floor - previousFloor)
        previousFloor = floor + 1L
    }

    maxLength = max(top.toLong() - previousFloor + 1, maxLength)
    return maxLength.toInt()
}

fun longestConsecutive(nums: IntArray): Int {

    val set = nums.toMutableSet()
    val length = set.size
    var maxLength = 0
    while (set.isNotEmpty()) {
        val num = set.first()
        set.remove(num)
        var left = 1
        while ((num - left) in set) {
            set.remove(num - left)
            left++
        }
        var right = 1
        while ((num + right) in set) {
            set.remove(num + right)
            right++
        }
        maxLength = max(maxLength, left + right - 1)
        if (maxLength > length / 2) return maxLength
    }
    return maxLength

}

fun longestContinuousSubstring(s: String): Int {
    val n = s.length

    val dp = IntArray(n)
    dp[0] = 1

    for (i in 1 until n) {
        if (s[i].code == s[i - 1].code + 1) {
            dp[i] = dp[i - 1] + 1
        } else {
            dp[i] = 1
        }

    }
    return dp.max()
}

fun numberOfArithmeticSlices(nums: IntArray): Int {
    val n = nums.size
    if (n < 3) return 0
    val dp = IntArray(n)
    dp[0] = 0
    dp[1] = 0

    for (i in 2 until n) {
        if (nums[i] - nums[i - 1] == nums[i - 1] - nums[i - 2]) {
            dp[i] = dp[i - 1] + 1
        } else {
            dp[i] = 0
        }
    }
    // println(dp.toList())
    return dp.sum()
}

fun zeroFilledSubarray(nums: IntArray): Long {
    val n = nums.size
    val dp = DoubleArray(n)
    dp[0] = if (nums[0] == 0) 1.0 else 0.0
    for (i in 1 until n) {
        if (nums[i] == 0) {
            dp[i] = dp[i - 1] + 1.0
        } else {
            dp[i] = 0.0
        }
    }
    // println(dp.toList())
    return dp.sum().toLong()
}

fun getDescentPeriods(prices: IntArray): Long {
    val n = prices.size
    val dp = DoubleArray(n)
    dp[0] = 1.0
    for (i in 1 until n) {
        if (prices[i] == prices[i - 1] - 1) {
            dp[i] = dp[i - 1] + 1.0
        } else {
            dp[i] = 1.0
        }
    }
    // println(dp.toList())
    return dp.sum().toLong()

}

fun partition(head: ListNode?, x: Int): ListNode? {
    head ?: return null
    val beforeHead = ListNode(0)
    val afterHead = ListNode(0)

    var before: ListNode? = beforeHead
    var after: ListNode? = afterHead
    var node: ListNode? = head
    while (node != null) {
        if (node.`val` < x) {
            before?.next = node
            before = before?.next
        } else {
            after?.next = node
            after = after?.next
        }
        node = node.next
    }
    after?.next = null
    before?.next = afterHead.next
    return beforeHead.next
}

fun subsetsWithDup(nums: IntArray): List<List<Int>> {
    val n = nums.size
    val subSetCount = (2 shl n) - 1
    val result = mutableSetOf<List<Int>>()
    for (bit in 0 until subSetCount) {
        val subSet = mutableListOf<Int>()
        for (i in 0 until n) {
            if ((bit shr i) and 1 == 1) {
                subSet.add(nums[i])
            }
        }
        result.add(subSet.sorted())
    }

    return result.toList()
}

fun eventualSafeNodes(graph: Array<IntArray>): List<Int> {
    val n = graph.size
    val visited = mutableSetOf<Int>()
    val safeNodes = mutableSetOf<Int>()
    val terminalNodes = mutableSetOf<Int>()
    val done = mutableSetOf<Int>()
    fun dfs(u: Int): Boolean {
        visited.add(u)
        done.add(u)

        if (graph[u].isEmpty()) {
            terminalNodes.add(u)
            return true
        }
        var leadsToTerminal = true
        for (v in graph[u]) {
            if (v !in visited) {
                leadsToTerminal = leadsToTerminal && dfs(v)
                visited.remove(v)
            } else if (v !in terminalNodes && v !in safeNodes) {
                leadsToTerminal = false
            }
        }
        if (leadsToTerminal) {
            safeNodes.add(u)
        }
        return leadsToTerminal
    }

    for (i in 0 until n) {
        if (i !in visited && i !in done) {
            dfs(i)
        }
    }
    return (safeNodes + terminalNodes).sorted()
}

fun findCircleNum(isConnected: Array<IntArray>): Int {
    val n = isConnected.size
    val graph = Array(n) { mutableListOf<Int>() }
    val parent = IntArray(n)
    val visited = mutableSetOf<Int>()

    for (i in 0 until n) {
        for (j in 0 until n) {
            if (isConnected[i][j] == 1) {
                graph[i].add(j)
                graph[j].add(i)
            }
        }
    }

    fun dfs(u: Int) {
        visited.add(u)
        for (v in graph[u]) {
            if (v != parent[u] && v !in visited) {
                parent[v] = u
                dfs(v)
            }
        }
    }

    var cnt = 0
    for (i in 0 until n) {
        if (i !in visited) {
            dfs(i)
            cnt++
        }
    }
    return cnt
}


fun canSeePersonsCount(heights: IntArray): IntArray {
    val numbers = heights.toList()
    val n = numbers.size
    val stack = Stack<Int>()

    println(heights.indices.toList())
    println(heights.toList())

    val result = IntArray(n)

    for (i in 0 until n) {
        var hiddenCount = 0
        val height = numbers[i]
        while (stack.isNotEmpty() && height >= numbers[stack.peek()]) {
            val start = stack.pop()
            val end = if (i == n - 1) n - 1 else i
            result[start] = end - start - hiddenCount
            hiddenCount += (result[start] - 1)
        }
        stack.push(i)
    }
    var hiddenCount = 0
    while (stack.isNotEmpty()) {
        val top = stack.pop()
        result[top] = n - 1 - top - hiddenCount
        hiddenCount += result[top]
    }

    return result
}

fun numTeams(rating: IntArray): Int {
    val n = rating.size

    var count = 0

    for (i in 1..(n - 2)) {
        val num = rating[i]
        var lessLeft = 0
        var greaterLeft = 0
        var lessRight = 0
        var greaterRight = 0

        for (l in 0 until i) {
            if (rating[l] < num) lessLeft++
            if (rating[l] > num) greaterLeft++
        }

        for (r in (n - 1) downTo (i + 1)) {
            if (rating[r] < num) lessRight++
            if (rating[r] > num) greaterRight++
        }

        if (lessLeft > 0 && greaterRight > 0) {
            count += lessLeft * greaterRight
        }
        if (greaterLeft > 0 && lessRight > 0) {
            count += greaterLeft * lessRight
        }
    }
    return count
}

fun isMonotonic(nums: IntArray): Boolean {
    val n = nums.size
    val increasedList = nums.sorted()
    val decreasedList = nums.sortedDescending()

    var isIncrease = true
    for (i in 0 until n) {
        if (nums[i] != increasedList[i]) {
            isIncrease = false
            break
        }
    }

    var isDecrease = true
    for (i in 0 until n) {
        if (nums[i] != decreasedList[i]) {
            isDecrease = false
            break
        }
    }


    return isIncrease || isDecrease
}

fun reversePairs(nums: IntArray): Int {
    val n = nums.size
    val numbers = DoubleArray(n) { nums[it].toDouble() }
    val tree = RightFloatSmallerSegmentTree(numbers)

    var count = 0
    for (i in 0 until n - 1) {
        val base = numbers[i] * 0.5
        count += tree.getSmallerCount(base, i + 1, n - 1)
    }
    return count
}

fun countFairPairs(nums: IntArray, lower: Int, upper: Int): Long {
    val n = nums.size
    val numbers = LongArray(n) { nums[it].toLong() }
    val tree = RightSumSegmentTree(numbers)
    var count = 0

    for (i in 0 until n - 1) {
        val num = nums[i]

        count += tree.countInRange(lower.toLong() - num, upper.toLong() - num, i + 1, n - 1)
    }
    return count.toLong()
}

class FairPairSegmentTree(private val data: IntArray) {
    private val n = data.size
    private val tree = Array(4 * n) { emptyList<Int>() }

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = listOf(data[l])
        } else {
            val mid = (l + r) / 2
            build(node * 2, l, mid)
            build(node * 2 + 1, mid + 1, r)
            val left = tree[node * 2]
            val right = tree[node * 2 + 1]
            tree[node] = merge(left, right)
        }
    }

    private fun merge(left: List<Int>, right: List<Int>): List<Int> {
        val m = left.size
        val n = right.size
        var i = 0
        var j = 0
        return buildList {
            while (i < m || j < n) {
                if (i >= m) {
                    add(right[j++])
                    continue
                }
                if (j >= n) {
                    add(left[i++])
                    continue
                }
                val a = left[i]
                val b = right[j]
                if (a <= b) {
                    add(a)
                    i++
                } else {
                    add(b)
                    j++
                }
            }
        }

    }

    fun countInRange(lower: Int, upper: Int, start: Int, end: Int): Int {
        return query(1, 0, n - 1, lower, upper, start, end)
    }

    private fun query(node: Int, l: Int, r: Int, lower: Int, upper: Int, start: Int, end: Int): Int {
        if (r < start || l > end) return 0

        if (start <= l && r <= end) {
            val lowerBound = lowerBoundInRange(tree[node], lower, 0, tree[node].size - 1)
            if (lowerBound < 0) return 0
            val upperBound = upperBoundInRange(tree[node], upper, 0, tree[node].size - 1)
            if (upperBound < 0) return 0
            return upperBound - lowerBound + 1
        }

        val mid = (l + r) / 2
        val left = query(node * 2, l, mid, lower, upper, start, end)
        val right = query(node * 2 + 1, mid + 1, r, lower, upper, start, end)
        return left + right
    }


    private fun lowerBoundInRange(arr: List<Int>, lower: Int, l: Int, r: Int): Int {
        var left = l
        var right = r
        var res = -1
        while (left <= right) {
            val mid = left + (right - left) / 2
            if (arr[mid] >= lower) {
                res = mid
                right = mid - 1
            } else {
                left = mid + 1
            }
        }
        return res
    }

    private fun upperBoundInRange(arr: List<Int>, upper: Int, l: Int, r: Int): Int {
        var left = l
        var right = r
        var res = -1
        while (left <= right) {
            val mid = left + (right - left) / 2
            if (arr[mid] <= upper) {
                res = mid
                left = mid + 1
            } else {
                right = mid - 1
            }
        }
        return res
    }
}

fun countRangeSum(nums: IntArray, lower: Int, upper: Int): Int {
    val n = nums.size
    var temporarySum = 0L
    val sums = LongArray(n + 1) {
        if (it == 0) 0 else {
            temporarySum += nums[it - 1]
            temporarySum
        }
    }
    //  println(nums.toList())
    //  println(sums.toList())

    val tree = RightSumSegmentTree(sums)


    var count = 0

    for (i in 0..n) {
        val sum = sums[i]

        count += tree.countInRange(lower + sum, upper + sum, i + 1, n)
    }
    return count
}


fun countHillValley(nums: IntArray): Int {
    val numbers = mutableListOf<Int>()

    for (num in nums) {
        if (num != numbers.lastOrNull()) numbers.add(num)
    }
    val n = numbers.size
    var count = 0
    for (i in 1 until n - 1) {
        val num = numbers[i]
        val previous = numbers[i - 1]
        val next = numbers[i + 1]
        if (num > previous && num > next) count++
        if (num < previous && num < next) count++

    }
    return count
}

fun countSmaller(nums: IntArray): List<Int> {
    val n = nums.size
    val tree = RightSmallerSegmentTree(nums)
    return List(n) {
        if (it == n - 1) 0 else tree.getSmallerCount(nums[it], it + 1, n - 1)
    }
}

fun minimumJumps(forbidden: IntArray, a: Int, b: Int, x: Int): Int {
    val forbiddenSet = forbidden.toSet()

    val limitStep = maxOf(x + a + b, forbidden.max() + a + b)

    var minStepCount = Int.MAX_VALUE
    val visited = mutableSetOf<Pair<Int, Boolean>>()

    val queue = PriorityQueue<Triple<Int, Boolean, Int>>(compareBy { (pos, _, step) ->
        step + abs(pos - x) / a
    })
    queue.add(Triple(0, false, 0))
    while (queue.isNotEmpty()) {
        val state = queue.poll()
        val (i, isForwarded, step) = state
        visited.add(i to isForwarded)
        if (i == x) {
            minStepCount = min(minStepCount, step)
            return minStepCount
        }
        if (step >= minStepCount) continue
        val forwardStep = i + a
        val shouldForward = forwardStep <= limitStep &&
                forwardStep to true !in visited && forwardStep !in forbiddenSet

        if (shouldForward) {
            val newState = Triple(forwardStep, true, step + 1)
            queue.add(newState)
            visited.remove(forwardStep to true)
        }

        val backwardStep = i - b
        val shouldBackward = backwardStep >= 0 &&
                isForwarded &&
                backwardStep to false !in visited &&
                backwardStep !in forbiddenSet
        if (shouldBackward) {
            val newState = Triple(backwardStep, false, step + 1)
            queue.add(newState)
            visited.remove(backwardStep to false)
        }
    }

    return -1
}

fun maximumJumps(nums: IntArray, target: Int): Int {
    val n = nums.size
    if (n == 2) {
        return if (abs(nums[0] - nums[1]) <= target) 1 else -1
    }

    val dp = IntArray(n) { -1 }
    dp[0] = 0

    val segmentTree = MaxSegmentTree(nums)

    for (i in 1 until n) {

        val maxStepsSoFar = segmentTree.getMax(0, i - 1)
        dp[i] = maxStepsSoFar + 1
        segmentTree.update(i, dp[i])
    }

    return dp[n - 1]
}

fun maximumScore(nums: IntArray, k: Int): Int {
    val n = nums.size
    //  println(heights.toList())
    val stack = Stack<Int>()

    val left = IntArray(n)

    for (i in (n - 1) downTo 0) {
        while (stack.isNotEmpty() && nums[i] < nums[stack.peek()]) {
            left[stack.pop()] = i
        }
        stack.push(i)
    }

    while (stack.isNotEmpty()) {
        left[stack.pop()] = -1
    }

    // println(left.toList())


    val right = IntArray(n)
    for (i in 0 until n) {
        while (stack.isNotEmpty() && nums[i] < nums[stack.peek()]) {
            right[stack.pop()] = i
        }
        stack.push(i)
    }

    while (stack.isNotEmpty()) {
        right[stack.pop()] = n
    }
    // println(right.toList())

    var maxScore = 0

    for (i in 0 until n) {
        if (left[i] + 1 <= k && right[i] - 1 >= k) {
            val score = nums[i] * (right[i] - left[i] - 1)
            maxScore = max(maxScore, score)
        }
    }

    return maxScore

}

fun maxSubarrays(n: Int, conflictingPairs: Array<IntArray>): Long {
    for (pair in conflictingPairs) {
        pair.sort()
    }

    val lToR = TreeMap<Int, MutableList<Int>>()
    for ((l, r) in conflictingPairs) {
        lToR.computeIfAbsent(l) { mutableListOf() }.add(r)
    }
    for (list in lToR.values) list.sort()

    val f = IntArray(n + 1) { n }
    f[0] = 0
    val conflictMap = Array(n + 1) { mutableSetOf<Int>() }

    for ((a, b) in conflictingPairs) {
        conflictMap[a].add(b)
        conflictMap[b].add(a)
    }

    for (i in 1..n) {
        val selected = mutableSetOf<Int>()
        var bestEnd = n
        for (j in i..n) {
            if (conflictMap[j].any { it in selected }) continue
            selected.add(j)
            bestEnd = minOf(bestEnd, j)
        }

        f[i] = if (selected.isEmpty()) i else selected.maxOrNull()!!
    }

    val result = f.sumOf { it.toLong() } - n.toLong() * (n + 1) / 2 + n.toLong()

    println(f.toList())
    return result
}


fun maximalRectangle(matrix: Array<CharArray>): Int {
    val m = matrix.size
    val n = matrix[0].size

    val histogram = IntArray(n)

    var maxArea = 0
    for (i in 0 until m) {

        for (j in 0 until n) {
            if (matrix[i][j] == '0') {
                histogram[j] = 0
            } else {
                histogram[j]++
            }
        }
        maxArea = max(maxArea, largestRectangleArea(histogram))
    }
    //  println(matrix.joinToString("\n") { it.toList().toString() })
    return maxArea
}

fun largestRectangleArea(heights: IntArray): Int {
    val n = heights.size
    //  println(heights.toList())
    val stack = Stack<Int>()

    val left = IntArray(n)

    for (i in (n - 1) downTo 0) {
        while (stack.isNotEmpty() && heights[i] < heights[stack.peek()]) {
            left[stack.pop()] = i
        }
        stack.push(i)
    }

    while (stack.isNotEmpty()) {
        left[stack.pop()] = -1
    }

    // println(left.toList())


    val right = IntArray(n)
    for (i in 0 until n) {
        while (stack.isNotEmpty() && heights[i] < heights[stack.peek()]) {
            right[stack.pop()] = i
        }
        stack.push(i)
    }

    while (stack.isNotEmpty()) {
        right[stack.pop()] = n
    }
    // println(right.toList())

    var maxArea = 0

    for (i in 0 until n) {
        if (heights[i] > 0) {
            val area = heights[i] * (right[i] - left[i] - 1)
            maxArea = max(maxArea, area)
        }
    }

    return maxArea

}

fun countIncreasingSubsequencesK(nums: IntArray, k: Int): Long {
    val n = nums.size
    val maxVal = nums.max()

    val tree = Array(k) {
        SumLongSegmentTree(List(maxVal + 1) { 0L })
    }

    for (num in nums) {
        tree[0].update(num, 1)

        for (len in 1 until k) {
            val count = tree[len - 1].sumRange(1, num - 1)
            if(count > 0) tree[len].update(num, count)
        }
    }

    for (i in 0 until k) {
        println((0 until n).map {
            tree[i].sumRange(nums[it], nums[it])
        })
    }
    return tree[k - 1].sumRange(1, maxVal)
}
