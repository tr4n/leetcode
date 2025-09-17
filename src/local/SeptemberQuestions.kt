package local

import java.util.PriorityQueue
import java.util.Stack
import java.util.TreeMap
import java.util.TreeSet
import kotlin.math.cbrt
import kotlin.math.sqrt

fun areaOfMaxDiagonal(dimensions: Array<IntArray>): Int {
    val (width, length) = dimensions.maxWith(
        comparator = compareBy<IntArray> { it[0] * it[0] + it[1] * it[1] }
            .thenBy { it[0] * it[1] }
    )
    return width * length
}

fun sumOfFlooredPairs(nums: IntArray): Int {
    val mod = 1_000_000_007
    val n = nums.size
    var minValue = Int.MAX_VALUE
    var maxValue = Int.MIN_VALUE

    val distinctNumbers = mutableSetOf<Int>()
    for (num in nums) {
        distinctNumbers.add(num)
        minValue = minOf(minValue, num)
        maxValue = maxOf(maxValue, num)
    }

    val freq = IntArray(maxValue + 1)
    for (num in nums) freq[num]++
    val prefix = LongArray(maxValue + 1)
    for (i in 1..maxValue) prefix[i] = prefix[i - 1] + freq[i].toLong()

    var sum = 0L
    for (num in distinctNumbers) {
        val maxMultiple = maxValue / num

        // sum = (sum + num * freq[num]) % mod
        //   sum = (sum + prefix[minOf(2 * num - 1, maxValue)] - prefix[num]) % mod
        for (i in 1..maxMultiple) {
            val left = num * i
            val right = (num * (i + 1) - 1).coerceIn(left, maxValue)
            val total = (prefix[right] - prefix[left - 1]) * freq[num] * i
            //  println("$left $right ${freq[num]} $total")
            sum += total
            sum %= mod
        }
    }
    return (sum % mod).toInt()
}

fun minWastedSpace(packages: IntArray, boxes: Array<IntArray>): Int {
    val mod = 1_000_000_007
    var minValue = Int.MAX_VALUE
    var maxValue = Int.MIN_VALUE

    for (num in packages) {
        minValue = minOf(minValue, num)
        maxValue = maxOf(maxValue, num)
    }


    val freq = IntArray(maxValue + 1)
    for (num in packages) freq[num]++

    val prefixFreq = LongArray(maxValue + 1)
    val prefixSum = LongArray(maxValue + 1)
    for (i in minValue..maxValue) {
        prefixFreq[i] = prefixFreq[i - 1] + freq[i].toLong()
        prefixSum[i] = prefixSum[i - 1] + freq[i].toLong() * i
    }

    boxes.onEach { it.sort() }
    val availableBoxes = boxes.filter { it.last() >= maxValue }

    var minWastedSpace = Long.MAX_VALUE
    for (boxList in availableBoxes) {
        var previousBox = 0
        var wasted = 0L
        for (box in boxList) {
            val current = box.coerceAtMost(maxValue)
            val previous = previousBox.coerceAtMost(maxValue)

            val count = prefixFreq[current] - prefixFreq[previous]
            val packageSum = prefixSum[current] - prefixSum[previous]
            val totalSpace = count * box
            wasted += (totalSpace - packageSum)
            previousBox = box
            //    println("$box $count $packageSum")
            if (wasted >= minWastedSpace) break
        }
        minWastedSpace = minOf(minWastedSpace, wasted)
    }
    return if (minWastedSpace == Long.MAX_VALUE) -1 else (minWastedSpace % mod).toInt()
}

fun findMedianSortedArrays(nums1: IntArray, nums2: IntArray): Double {
    fun findPos(arr1: IntArray, arr2: IntArray, k: Int): Int {
        val m = arr1.size
        val n = arr2.size

        if (m > n) {
            return findPos(arr2, arr1, k)
        }

        var lo = maxOf(0, k - n)
        var hi = minOf(k, m)

        while (lo <= hi) {
            val cut1 = (lo + hi) / 2
            if (cut1 > k) {
                hi = cut1 - 1
                continue
            }

            val cut2 = k - cut1
            val left1 = if (cut1 == 0) Int.MIN_VALUE else arr1[cut1 - 1]
            val right1 = if (cut1 == m) Int.MAX_VALUE else arr1[cut1]

            val left2 = if (cut2 == 0) Int.MIN_VALUE else arr2[cut2 - 1]
            val right2 = if (cut2 == n) Int.MAX_VALUE else arr2[cut2]

            when {
                left1 <= right2 && left2 <= right1 -> return maxOf(left1, left2)
                left1 > right2 -> hi = cut1 - 1
                else -> lo = cut1 + 1
            }

        }
        return Int.MIN_VALUE
    }

    val m = nums1.size
    val n = nums2.size
    val total = m + n

    return if (total % 2 != 0) {
        findPos(nums1, nums2, total / 2 + 1).toDouble()
    } else {
        val mid1 = findPos(nums1, nums2, total / 2).toDouble()
        val mid2 = findPos(nums1, nums2, 1 + total / 2).toDouble()
        (mid1 + mid2) / 2.0
    }
}

fun isValidSudoku(board: Array<CharArray>): Boolean {

    for (i in 0 until 9) {
        var rowMask = 0
        var colMask = 0
        for (j in 0 until 9) {
            val cRow = board[i][j]
            if (cRow in '0'..'9') {
                val num = cRow - '0'
                val bit = 1 shl num
                if (rowMask and bit != 0) return false
                rowMask = rowMask or bit
            }
            val cCol = board[j][i]
            if (cCol in '0'..'9') {
                val num = cCol - '0'
                val bit = 1 shl num
                if (colMask and bit != 0) return false
                colMask = colMask or bit
            }
        }
    }

    for (br in 0 until 3) {          // block row
        for (bc in 0 until 3) {      // block col
            var mask = 0
            for (r in br * 3 until br * 3 + 3) {
                for (c in bc * 3 until bc * 3 + 3) {
                    val ch = board[r][c]
                    if (ch !in '0'..'9') continue
                    val num = ch - '0'
                    val bit = 1 shl num
                    if (mask and bit != 0) return false
                    mask = mask or bit
                }
            }
        }
    }
    return true
}

fun getLeastFrequentDigit(n: Int): Int {
    return n.toString().groupingBy { it }.eachCount()
        .toList()
        .sortedWith(compareBy<Pair<Char, Int>> { it.second }.thenBy { it.first - '0' })
        .first()
        .first - '0'
}

//fun score(cards: Array<String>, x: Char): Int {
//    val map = cards.groupBy { card ->
//        when {
//            card[0] == x && card[1] != x -> 0
//            card[0] != x && card[1] == x -> 1
//            card[0] == x && card[1] == x -> 2
//            else -> 3
//        }
//    }
//    val firstList = map[0] ?: emptyList()
//    val secondList = map[1] ?: emptyList()
//    val bothList = map[2] ?: emptyList()
//
//    fun countMaxPairs(list: List<String>, pos: Int): Pair<Int, Int> {
//        val freq = IntArray(30)
//        for (pair in list) {
//            freq[pair[pos] - 'a']++
//        }
//        val total = list.size
//        val maxFreq = freq.max()
//        val maxPairs = minOf(total / 2, total - maxFreq)
//        val remaining = total - 2 * maxPairs
//        return maxPairs to remaining
//    }
//
//
//    val (a, firstRemain) = countMaxPairs(firstList, 1)
//    val (b, secondRemain) = countMaxPairs(secondList, 0)
//    val bothCount = bothList.size
//    println(firstList.size.toString() + " " + firstList)
//    println(secondList.size.toString() + " " + secondList)
//    println("$a $b $bothCount ($firstRemain, $secondRemain)")
//    val c = minOf(bothCount, firstRemain + secondRemain)
//    return a + b + c
//}
fun score(cards: Array<String>, x: Char): Int {
    val map = cards.groupBy { card ->
        when {
            card[0] == x && card[1] != x -> 0
            card[0] != x && card[1] == x -> 1
            card[0] == x && card[1] == x -> 2
            else -> 3
        }
    }
    val firstList = map[0] ?: emptyList()
    val secondList = map[1] ?: emptyList()
    val bothList = map[2] ?: emptyList()

    fun countMaxPairs(list: List<String>, pos: Int): Pair<Int, Int> {
        val freq = IntArray(26)
        for (pair in list) {
            freq[pair[pos] - 'a']++
        }
        val total = list.size
        val maxFreq = freq.maxOrNull() ?: 0
        val maxPairs = minOf(total / 2, total - maxFreq)
        return Pair(total, maxPairs)
    }

    val (totalFirst, maxPairsFirst) = countMaxPairs(firstList, 1)
    val (totalSecond, maxPairsSecond) = countMaxPairs(secondList, 0)
    val bothCount = bothList.size

    var best = 0

    for (p1 in 0..maxPairsFirst) {
        val rem1 = totalFirst - 2 * p1

        var curBestSide2 = 0
        val totalR = totalSecond
        val maxR = maxPairsSecond
        val t = (rem1 + totalR) - bothCount
        val candidates = listOf(0, maxR, t / 2, t / 2 + 1)
        for (c in candidates) {
            val p2 = c.coerceIn(0, maxR)
            val rem2 = totalR - 2 * p2
            if (rem2 < 0) continue
            val useXX = minOf(bothCount, rem1 + rem2)
            if (useXX < 0) continue
            curBestSide2 = maxOf(curBestSide2, p2 + useXX)
        }
        best = maxOf(best, p1 + curBestSide2)
    }
    return best
}

fun uniquePaths(grid: Array<IntArray>): Int {
    val mod = 1_000_000_007
    val m = grid.size
    val n = grid[0].size
    val dp = Array(m) { LongArray(n) }
    dp[0][0] = 1

    fun findNext(i: Int, j: Int, dir: Int): Pair<Int, Int>? {
        if (i !in 0 until m || j !in 0 until n) return null
        if (grid[i][j] == 0) return i to j
        return if (dir == 0) {
            findNext(i + 1, j, 1)
        } else {
            findNext(i, j + 1, 0)
        }
    }
    for (i in 0 until m) {
        for (j in 0 until n) {
            if (dp[i][j] == 0L) continue
            if (i + 1 < m) {
                val next = findNext(i + 1, j, 1)
                if (next != null) {
                    val (ni, nj) = next
                    dp[ni][nj] += dp[i][j]
                    dp[ni][nj] %= mod
                }
            }
            if (j + 1 < n) {
                val next = findNext(i, j + 1, 0)
                if (next != null) {
                    val (ni, nj) = next
                    dp[ni][nj] += dp[i][j]
                    dp[ni][nj] %= mod
                }
            }
        }
    }
    return (dp[m - 1][n - 1] % mod).toInt()
}

fun solveSudoku(board: Array<CharArray>): Unit {
    val filledRow = IntArray(9)
    val filledCol = IntArray(9)
    val filledBlock = IntArray(9)
    val blank = mutableListOf<Pair<Int, Int>>()

    for (r in 0 until 9) {
        for (c in 0 until 9) {
            if (board[r][c] in '1'..'9') {
                val num = board[r][c] - '0'
                val bit = 1 shl num
                filledRow[r] = filledRow[r] or bit
                filledCol[c] = filledCol[c] or bit
                val blockId = (r / 3) * 3 + (c / 3)
                filledBlock[blockId] = filledBlock[blockId] or bit
                num
            } else {
                blank.add(r to c)
                0
            }
        }
    }

    fun candidate(r: Int, c: Int): List<Int> {
        val list = mutableListOf<Int>()
        val rowMask = filledRow[r]
        val colMask = filledCol[c]
        val blockMask = filledBlock[(r / 3) * 3 + (c / 3)]
        val mask = rowMask or colMask or blockMask
        for (num in 1..9) {
            val bit = 1 shl num
            if (mask and bit != 0) continue
            list.add(num)
        }
        return list
    }

    blank.sortBy { (r, c) -> candidate(r, c).size }

    fun fill(r: Int, c: Int, num: Int, enable: Boolean) {
        if (enable) {
            val bit = 1 shl num
            filledRow[r] = filledRow[r] or bit
            filledCol[c] = filledCol[c] or bit
            val blockId = (r / 3) * 3 + (c / 3)
            filledBlock[blockId] = filledBlock[blockId] or bit
            board[r][c] = num.digitToChar()
        } else {
            val bit = (1 shl num).inv()
            filledRow[r] = filledRow[r] and bit
            filledCol[c] = filledCol[c] and bit
            val blockId = (r / 3) * 3 + (c / 3)
            filledBlock[blockId] = filledBlock[blockId] and bit
            board[r][c] = '.'
        }
    }

    var found = false
    fun backtrack(pos: Int) {
        if (found) return
        if (pos == blank.size) {
            found = true
            return
        }
        val (r, c) = blank[pos]
        val candidates = candidate(r, c)
        for (num in candidates) {
            fill(r, c, num, true)
            backtrack(pos + 1)
            if (found) return
            fill(r, c, num, false)
        }
    }
    backtrack(0)
    println(board.joinToString("\n") { String(it) })
}

fun recoverOrder(order: IntArray, friends: IntArray): IntArray {
    val set = friends.toSet()
    return order.filter { it in friends }.toIntArray()
}


fun maxProduct(nums: IntArray): Long {
    class Node {
        val children = arrayOfNulls<Node>(2)
        var maxValue = 0
    }

    val root = Node()

    for (num in nums) {
        var node = root
        node.maxValue = maxOf(node.maxValue, num)
        for (bit in 31 downTo 0) {
            val b = (num shr bit) and 1
            if (node.children[b] == null) node.children[b] = Node()
            node = node.children[b]!!
            node.maxValue = maxOf(node.maxValue, num)
        }
    }
    var maxProduct = 0L
    println(nums.toList())
    for (first in nums) {
        var node = root
        var notFound = false

        for (bit in 31 downTo 0) {
            val firstBit = 1 and (first shr bit)
            val oppositeBit = 1 - firstBit
            val oppositeChild = node.children[oppositeBit]
            val originalChild = node.children[firstBit]
            if (oppositeChild != null) {
                node = oppositeChild
            } else if (originalChild != null) {
                node = originalChild
            } else {
                notFound = true
                break
            }

//            val child0 = node.children[0]
//            val child1 = node.children[1]
//            when {
//                firstBit == 1 -> {
//                    if (child0 == null) {
//                        notFound = true
//                        break
//                    }
//                    node = child0
//                }
//
//                child1 != null -> {
//                    node = child1
//                }
//
//                child0 != null -> {
//                    node = child0
//                }
//
//                else -> {
//                    notFound = true
//                    break
//                }
//            }
        }
        if (notFound) continue
        val second = node.maxValue
        if (first and second != 0) continue
        println("$first $second")
        maxProduct = maxOf(maxProduct, 1L * second * first)
    }

    return maxProduct
}

fun minDifference(n: Int, k: Int): IntArray {

    fun find2(m: Int): IntArray {
        var first = 1
        for (num in sqrt(m.toDouble()).toInt() downTo 1) {
            if (m % num == 0) {
                first = num
                break
            }
        }
        val second = m / first
        return intArrayOf(first, second)
    }
    if (k == 2) return find2(n)
    if (k == 3) {
        val cube = cbrt(n.toDouble()).toInt()
        var bestA = 1
        var minDiff = Int.MAX_VALUE
        for (a in cube downTo 1) {
            if (n % a == 0) {
                val m = n / a
                val pair = find2(m)
                val temp = intArrayOf(a, pair[0], pair[1])
                temp.sort()
                val diff = temp[2] - temp[0]
                if (diff < minDiff) {
                    minDiff = diff
                    bestA = a
                }
            }
        }
        val m = n / bestA
        val best2 = find2(m)
        val result = intArrayOf(best2[0], best2[1], bestA)
        result.sortDescending()
        return result
    }

    if (k == 4) {
        val fourth = Math.pow(n.toDouble(), 0.25).toInt()
        var bestA = 1
        for (a in fourth downTo 1) {
            if (n % a == 0) {
                bestA = a
                break
            }
        }
        val rest = minDifference(n / bestA, 3)
        return intArrayOf(bestA) + rest
    }

    // k = 5
    if (k == 5) {
        val fifth = Math.pow(n.toDouble(), 0.2).toInt()
        var bestA = 1
        for (a in fifth downTo 1) {
            if (n % a == 0) {
                bestA = a
                break
            }
        }
        val rest = minDifference(n / bestA, 4)
        return intArrayOf(bestA) + rest
    }


    val factors = mutableListOf<Int>()
    var x = n
    var d = 2
    while (d * d <= x) {
        while (x % d == 0) {
            factors.add(d)
            x /= d
        }
        d++
    }
    if (x > 1) factors.add(x)

    factors.sortDescending()
    //  println(factors.toList())
    val bucket = IntArray(k) { 1 }

    for (i in factors.indices) {
        var minValue = Int.MAX_VALUE
        var minIndex = -1
        for (i in 0 until k) {
            if (bucket[i] < minValue) {
                minValue = bucket[i]
                minIndex = i
            }
        }
        bucket[minIndex] *= factors[i]
        //  println(bucket.toList())
    }
    var balanced = true
    while (balanced) {
        balanced = false
        var maxIdx = 0
        var minIdx = 0
        var maxVal = bucket[0]
        var minVal = bucket[0]
        for (i in 1 until k) {
            if (bucket[i] > maxVal) {
                maxVal = bucket[i]
                maxIdx = i
            }
            if (bucket[i] < minVal) {
                minVal = bucket[i]
                minIdx = i
            }
        }

        for (d in listOf(2, 3, 5, 7, 11, 13)) {
            if (maxVal % d == 0 && maxVal / d > minVal) {
                bucket[maxIdx] /= d
                bucket[minIdx] *= d
                balanced = true
                break
            }
        }
    }
    bucket.sort()
    return bucket
}

fun maxAverageRatio(classes: Array<IntArray>, extraStudents: Int): Double {
    val pq = PriorityQueue<Pair<Int, Int>>(compareByDescending { (pass, total) ->
        val x = pass.toDouble()
        val y = total.toDouble()
        (y - x) / (y * (y + 1))
    })
    for (frac in classes) {
        pq.add(frac[0] to frac[1])
    }
    var k = extraStudents
    while (k > 0) {
        val (a, b) = pq.poll()
        pq.add(a + 1 to b + 1)
        k--
    }
    val sum = pq.sumOf { (pass, total) ->
        pass.toDouble() / total.toDouble()
    }
    return sum / pq.size
}

// 1 10 100
fun lexicalOrder(n: Int): List<Int> {
    val result = mutableListOf<Int>()

    fun dfs(num: Int) {
        result.add(num)
        for (i in 0..9) {
            val newNum = num * 10 + i
            if (newNum > n) break
            dfs(newNum)
        }
    }

    for (i in 1..9) {
        if (i > n) break
        dfs(i)
    }
    return result
}

fun lengthLongestPath(input: String): Int {
    // println(input)
    class Node(
        var name: String,
        var level: Int,
        var isFile: Boolean,
        var length: Int = 0,
    )

    val nodes = input.split("\n").map { line ->
        val name = line.substringAfterLast("\t")
        val level = line.count { it == '\t' }
        Node(
            name,
            level,
            name.contains("."),
        )
    }

    var maxLength = -1
    val stack = Stack<Node>()
    for (node in nodes) {
        while (stack.isNotEmpty() && stack.peek().level >= node.level) {
            stack.pop()
        }
        val parentLength = if (stack.isNotEmpty()) stack.peek().length + 1 else 0
        node.length = parentLength + node.name.length
        if (node.isFile) {
            maxLength = maxOf(maxLength, node.length)
        } else {
            stack.add(node)
        }
    }
    return if (maxLength == -1) 0 else maxLength
}

fun countDays(days: Int, meetings: Array<IntArray>): Int {
    val map = mutableMapOf<Int, Int>()
    meetings.sortBy { it[0] }
    for ((start, end) in meetings) {
        map[start] = (map[start] ?: 0) + 1
        map[end + 1] = (map[end + 1] ?: 0) - 1
    }
    var lastFree = -1
    val list = map.toList().sortedBy { it.first }
    var meetingCount = 0
    var freeDayCount = 0
    for ((day, delta) in map) {
        meetingCount += delta
        if (meetingCount != 0) {
            lastFree = -1
            continue
        }
        if (lastFree > 0) {
            freeDayCount += 0

        }
    }
    return 0

}

fun numberOfPairs(points: Array<IntArray>): Int {
    val tree = TreeMap<Int, MutableList<Int>>()

    for ((x, y) in points) {
        if (tree[x] == null) {
            tree[x] = mutableListOf(y)
        } else {
            tree[x]?.add(y)
        }
    }

    tree.forEach { it.value.sortDescending() }
    var cnt = 0
    for ((x, y) in points) {
        //  println("base: $x,$y")
        var minYSoFar = Int.MIN_VALUE
        for ((nextX, list) in tree.tailMap(x, true)) {
            var l = 0
            var r = list.size - 1
            var firstSmaller = Int.MIN_VALUE

            while (l <= r) {
                val mid = (l + r) / 2
                val value = list[mid]
                if (value < y || (nextX > x && value == y)) {
                    firstSmaller = value
                    r = mid - 1
                } else {
                    l = mid + 1
                }
            }
            if (firstSmaller == Int.MIN_VALUE || firstSmaller <= minYSoFar) continue
            //  println("$nextX, $firstSmaller")
            cnt++
            minYSoFar = firstSmaller
        }
    }
    return cnt
}

fun maximumSubarraySum(nums: IntArray, k: Int): Long {
    val n = nums.size
    val prefix = LongArray(n + 1)
    for (i in 0 until n) {
        prefix[i + 1] = prefix[i] + nums[i]
    }

    val map = mutableMapOf<Int, MutableList<Int>>()
    for (i in 0 until n) {
        val num = nums[i]
        if (map[num] == null) {
            map[num] = mutableListOf(i)
        } else {
            map[num]?.add(i)
        }
    }

    map.forEach { entry ->
        entry.value.sortBy { prefix[it + 1] }
    }

    return 0L

}

fun spiralOrder(matrix: Array<IntArray>): List<Int> {
    val m = matrix.size
    val n = matrix[0].size

    var x = 0
    var y = -1
    val used = Array(m) { BooleanArray(n) }
    val total = m * n
    val result = mutableListOf<Int>()
    // println(matrix.print())

    while (result.size < total) {
        // right
        while (y + 1 < n && !used[x][y + 1]) {
            y++
            result.add(matrix[x][y])
            used[x][y] = true
        }
        //down
        while (x + 1 < m && !used[x + 1][y]) {
            x++
            result.add(matrix[x][y])
            used[x][y] = true
        }

        // left
        while (y - 1 >= 0 && !used[x][y - 1]) {
            y--
            result.add(matrix[x][y])
            used[x][y] = true
        }

        //up
        while (x - 1 >= 0 && !used[x - 1][y]) {
            x--
            result.add(matrix[x][y])
            used[x][y] = true
        }
    }
    return result
}

fun generateMatrix(n: Int): Array<IntArray> {
    val m = n

    var x = 0
    var y = -1
    val matrix = Array(m) { IntArray(n) }
    val total = m * n
    var cnt = 0
    while (cnt < total) {
        // right
        while (y + 1 < n && matrix[x][y + 1] == 0) {
            y++
            matrix[x][y] = ++cnt
        }
        //down
        while (x + 1 < m && matrix[x + 1][y] == 0) {
            x++
            matrix[x][y] = ++cnt
        }

        // left
        while (y - 1 >= 0 && matrix[x][y - 1] == 0) {
            y--
            matrix[x][y] = ++cnt
        }

        //up
        while (x - 1 >= 0 && matrix[x - 1][y] == 0) {
            x--
            matrix[x][y] = ++cnt
        }
    }
    return matrix
}

fun spiralMatrix(m: Int, n: Int, head: ListNode?): Array<IntArray> {
    var x = 0
    var y = -1
    val matrix = Array(m) { IntArray(n) { -1 } }
    val total = m * n
    var node = head
    var cnt = 0
    while (cnt < total && node != null) {
        // right
        while (y + 1 < n && matrix[x][y + 1] == -1 && node != null) {
            y++
            matrix[x][y] = node.`val`
            ++cnt
            node = node.next
        }
        //down
        while (x + 1 < m && matrix[x + 1][y] == -1 && node != null) {
            x++
            matrix[x][y] = node.`val`
            ++cnt
            node = node.next
        }

        // left
        while (y - 1 >= 0 && matrix[x][y - 1] == -1 && node != null) {
            y--
            matrix[x][y] = node.`val`
            ++cnt
            node = node.next
        }

        //up
        while (x - 1 >= 0 && matrix[x - 1][y] == -1 && node != null) {
            x--
            matrix[x][y] = node.`val`
            ++cnt
            node = node.next
        }
    }
    return matrix
}

fun spiralMatrixIII(rows: Int, cols: Int, rStart: Int, cStart: Int): Array<IntArray> {
    val m = rows
    val n = cols

    var x = rStart
    var y = cStart

    val result = mutableListOf<IntArray>()
    val filled = mutableSetOf<Pair<Int, Int>>()

    fun addCell(x: Int, y: Int) {
        filled.add(x to y)
        if (x in 0 until m && y in 0 until n) {
            result.add(intArrayOf(x, y))
        }
    }
    addCell(x, y)
    y++

    while (result.size < m * n) {

        // right
        while (Pair(x + 1, y) in filled) {
            addCell(x, y)
            y++
        }
        if (result.size >= m * n) break

        // down
        while (Pair(x, y - 1) in filled) {
            addCell(x, y)
            x++
        }
        if (result.size >= m * n) break
        // left
        while (Pair(x - 1, y) in filled) {
            addCell(x, y)
            y--
        }
        if (result.size >= m * n) break
        // up
        while (Pair(x, y + 1) in filled) {
            addCell(x, y)
            x--
        }
        if (result.size >= m * n) break
    }

    return result.toTypedArray()
}

fun numSub(s: String): Int {
    val n = s.length
    var maxLen = 0
    var startIndex = -1

    fun power(base: Long, exp: Long, modulus: Long): Long {
        var res: Long = 1
        var b = base % modulus

        var e = exp
        while (e > 0) {
            if (e % 2 == 1L) {
                res = (res * b) % modulus
            }

            b = (b * b) % modulus
            e /= 2
        }
        return res
    }

    val mod = 1_000_000_007L
    var total = 0L
    for (i in 0 until n) {
        if (s[i] == '0' && startIndex >= 0) {
            val len = i - startIndex
            total = (total + (len * (len + 1)) / 2) % mod
            startIndex = -1
            continue
        }
        if (s[i] == '1' && startIndex == -1) {
            startIndex = i
        }
    }
    if (startIndex >= 0) {
        val len = n - startIndex
        total = (total + (len * (len + 1)) / 2) % mod
    }

    return total.toInt()
}

fun findMinFibonacciNumbers(k: Int): Int {
    if (k <= 3) return 1
    val fNums = TreeSet<Int>()
    fNums.add(1)
    fNums.add(2)
    fNums.add(3)
    var a = 2
    var b = 3
    while (b < k) {
        val num = a + b
        if (num == k) return 1
        fNums.add(num)
        a = b
        b = num
    }

    fun dp(num: Int): Int {
        if (num < 0) return Int.MAX_VALUE
        if (num in fNums) return 1
        val lower = fNums.lower(num) ?: return Int.MAX_VALUE
        return 1 + dp(num - lower)
    }

    return dp(k)
}

fun complexNumberMultiply(num1: String, num2: String): String {
    val (a1, b1) = num1.substringBeforeLast('i').split('+').map { it.toInt() }
    val (a2, b2) = num2.substringBeforeLast('i').split('+').map { it.toInt() }
    return "${a1 * a2 - b1 * b2}+${a1 * b2 + a2 * b1}i"
}

fun integerBreak(n: Int): Int {
    if (n == 2) return 1
    if (n == 3) return 2
    if (n == 4) return 4

    var k = 2
    var ans = 1
    while (true) {
        val num = n / k
        if (num <= 1) return ans
        val r = n % k
        var p = 1
        for (i in 0 until r) p *= (num + 1)
        for (i in 0 until (k - r)) p *= num
        ans = maxOf(ans, p)
        k++
    }
    return -1
}

fun optimalDivision(nums: IntArray): String {
    val n = nums.size
    if (n == 2) return "${nums[0]}/${nums[1]}"

    val builder = StringBuilder()
    builder.append("${nums[0]}/(")

    for (i in 1 until n - 1) {
        builder.append("${nums[i]}/")
    }
    builder.append("${nums[n - 1]})")
    return builder.toString()
}

fun maxFreqSum(s: String): Int {
    val freq = s.groupingBy { it }.eachCount().toList().sortedByDescending { it.second }
    var a = 0
    var b = 0
    for ((ch, cnt) in freq) {
        val isVowel = ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u'
        if (isVowel) {
            if (a == 0) a = cnt
        } else {
            if (b == 0) b = cnt
        }
        if (a > 0 && b > 0) return a + b
    }
    val n = 1
    return a + b
}

fun spellchecker(wordlist: Array<String>, queries: Array<String>): Array<String> {
    val vowerSet = setOf('a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U')
    val originalList = wordlist.toSet()
    fun lowerWord(word: String): String {
        val builder = StringBuilder()
        for (c in word) {
            if (c in vowerSet) {
                builder.append('_')
            } else {
                builder.append(c.lowercaseChar())
            }
        }
        return builder.toString()
    }

    val capCorrect = mutableMapOf<String, String>()
    val vowelCorrect = mutableMapOf<String, String>()

    for (word in wordlist) {
        val lowercase = word.lowercase()
        if (capCorrect[lowercase] == null) capCorrect[lowercase] = word
        val correctVowels = lowerWord(word)
        if (vowelCorrect[correctVowels] == null) vowelCorrect[correctVowels] = word
    }

    return Array(queries.size) {
        val query = queries[it]
        if (query in originalList) return@Array query
        capCorrect[query.lowercase()] ?: vowelCorrect[lowerWord(query)] ?: ""
    }
}

fun countStableSubsequences(nums: IntArray): Int {
    val mod = 1_000_000_007
    var endWith0 = 0L
    var endWith00 = 0L
    var endWith1 = 0L
    var endWith11 = 0L
    val n = nums.size

    for (i in 0 until n) {
        val num = nums[i] % 2
        if (num == 0) {
            val newEndWith00 = endWith0
            val newEndWith0 = 1L + endWith1 + endWith11
            endWith0 += newEndWith0
            endWith00 += newEndWith00
        } else {
            val newEndWith11 = endWith0
            val newEndWith1 = 1L + endWith00 + endWith0
            endWith1 += newEndWith1
            endWith11 += newEndWith11
        }
        endWith0 %= mod
        endWith00 %= mod
        endWith1 %= mod
        endWith11 %= mod
    }
    return ((endWith0 + endWith00 + endWith1 + endWith11) % mod).toInt()
}

fun main() {
//    println(
//        maxProduct(intArrayOf(9, 2, 19))
//    )
    println(

    )
}