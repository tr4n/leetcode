package remote

import local.to2DIntArray
import java.util.*
import kotlin.math.abs
import kotlin.math.ceil

fun findDuplicates(nums: IntArray): List<Int> {
    val n = nums.size
    val result = mutableListOf<Int>()
    for (i in 0 until n) {
        var currentNum = nums[i]
        while (currentNum != i + 1) {
            val correctPos = currentNum - 1
            if (nums[correctPos] == currentNum) break

            val temp = nums[correctPos]
            nums[correctPos] = currentNum
            nums[i] = temp
            currentNum = temp
        }

    }
    for (i in 0 until n) {
        if (nums[i] != i + 1) {
            result.add(nums[i])
        }
    }
    return result
}

fun waysToReachTarget(target: Int, types: Array<IntArray>): Int {
    val mod = 1_000_000_007
    val n = types.size

    val dp = Array(n) { LongArray(target + 1) }
    val (count0, mark0) = types[0]
    for (j in 0..count0) {
        val score = j * mark0
        if (score > target) break
        dp[0][score] = 1
    }
    for (i in 1 until n) {
        val (count, mark) = types[i]
        for (j in 0..count) {
            val score = j * mark
            for (p in score..target) {
                val d = dp[i - 1][p - score]
                dp[i][p] += d
                dp[i][p] = dp[i][p] % mod
            }
        }
    }
    // println(dp.print())
    return (dp[n - 1][target] % mod).toInt()
}

fun sumCounts(nums: IntArray): Int {
    val mod = 1_000_000_007
    val n = nums.size
    val lastSeen = mutableMapOf<Int, Int>()

    val dp = LongArray(n)
    dp[0] = 1L
    lastSeen[nums[0]] = 0
    var d = 1L
    var squares = 1L
    for (i in 1 until n) {
        val num = nums[i]
        val lastIndex = lastSeen[num] ?: -1
        lastSeen[num] = i
        d += (i - lastIndex).toLong()
        dp[i] = d
        squares += d
        squares %= mod
    }
    squares = (squares * squares) % mod
    println(dp.toList())
    return (squares % mod).toInt()
}

fun fullBloomFlowers(flowers: Array<IntArray>, people: IntArray): IntArray {
    flowers.sortBy { it[0] }
    val map = TreeMap<Int, Int>()
    for ((start, end) in flowers) {
        map[start] = (map[start] ?: 0) + 1
        map[end + 1] = (map[end + 1] ?: 0) - 1
    }
    var count = 0
    map.forEach { (time, delta) ->
        count += delta
        map[time] = count
    }
    // println(map.toList().joinToString("\n"))
    return IntArray(people.size) {
        val time = people[it]
        val count = map.floorEntry(time)?.value ?: 0
        maxOf(count, 0)
    }
}

fun minInterval(intervals: Array<IntArray>, queries: IntArray): IntArray {
    val queryList = queries.withIndex().sortedBy { it.value }
    //  intervals.sortWith(comparator = compareBy<IntArray> { it[1] }.thenByDescending { it[0] })
    val minQuery = queries.min()
    val maxQuery = queries.max()
    val intervalList = intervals.filter {
        it[0] <= maxQuery && it[1] >= minQuery
    }.sortedWith(compareBy({ it[0] }))

    val pq = PriorityQueue<Pair<Int, Int>>(compareBy { it.first })
    for ((start, end) in intervals) {
        if (start > maxQuery || end < minQuery) continue
        pq.add(start to end)
    }

    val answers = IntArray(queries.size) { -1 }
    var i = 0
    for (query in queryList) {

        while (pq.isNotEmpty()) {
            val (start, end) = pq.poll()

            if (query.value in start..end) {
                answers[query.index] = end - start + 1
                break
            } else i++
        }
    }
    return answers
}

fun lenOfVDiagonal(grid: Array<IntArray>): Int {
    val m = grid.size
    val n = grid[0].size
    val dr = intArrayOf(-1, -1, 1, 1)
    val dc = intArrayOf(-1, 1, -1, 1)
    val nextClockwise = intArrayOf(1, 3, 0, 2)

    val dp = Array(m) { Array(n) { Array(4) { IntArray(2) { -1 } } } }
    var maxLength = 0

    fun dfs(row: Int, col: Int, dir: Int, turned: Int): Int {
        if (dp[row][col][dir][turned] != -1) return dp[row][col][dir][turned]

        val current = grid[row][col]
        val nextNum = if (current < 2) 2 else 0

        var res = 1

        var nr = row + dr[dir]
        var nc = col + dc[dir]
        if (nr in 0 until m && nc in 0 until n && grid[nr][nc] == nextNum) {
            res = maxOf(res, 1 + dfs(nr, nc, dir, turned))
        }

        if (turned == 0) {
            val rotateDir = nextClockwise[dir]
            nr = row + dr[rotateDir]
            nc = col + dc[rotateDir]
            if (nr in 0 until m && nc in 0 until n && grid[nr][nc] == nextNum) {
                res = maxOf(res, 1 + dfs(nr, nc, rotateDir, 1))
            }
        }

        dp[row][col][dir][turned] = res
        return res
    }

    for (i in 0 until m) {
        for (j in 0 until n) {
            if (grid[i][j] == 1) {
                for (dir in 0 until 4) {
                    maxLength = maxOf(maxLength, dfs(i, j, dir, 0))
                }
            }
        }
    }

    return maxLength
}

fun sortMatrix(grid: Array<IntArray>): Array<IntArray> {
    //  println(grid.print())
    // println()
    val m = grid.size
    val n = grid[0].size
    for (d in -(n - 1)..<m) {
        val diagonal = mutableListOf<Int>()
        for (i in 0 until m) {
            val j = i - d
            if (j in 0 until n) {
                diagonal.add(grid[i][j])
            }
        }
        if (d < 0) diagonal.sort() else diagonal.sortDescending()
        var id = 0
        for (i in 0 until m) {
            val j = i - d
            if (j in 0 until n) {
                grid[i][j] = diagonal[id++]
            }
        }
    }
    return grid
}

fun diagonalSort(grid: Array<IntArray>): Array<IntArray> {
    //  println(grid.print())
    // println()
    val m = grid.size
    val n = grid[0].size
    for (d in -(n - 1)..<m) {
        val diagonal = mutableListOf<Int>()
        for (i in 0 until m) {
            val j = i - d
            if (j in 0 until n) {
                diagonal.add(grid[i][j])
            }
        }
        diagonal.sort()
        var id = 0
        for (i in 0 until m) {
            val j = i - d
            if (j in 0 until n) {
                grid[i][j] = diagonal[id++]
            }
        }
    }
    return grid
}

fun largestNumber(nums: IntArray): String {
    return nums
        .map { it.toString() }
        .sortedWith { a, b ->
            val ab = a + b
            val ba = b + a
            ba.compareTo(ab)
        }
        .joinToString("")
        .trimStart('0').ifEmpty { "0" }

}

class SegmentTree2D(private val points: List<Pair<Int, Int>>) {
    private val xs = points.map { it.first }.distinct().sorted()
    private val tree = Array(4 * xs.size) { emptyList<Int>() }

    init {
        build(1, 0, xs.size - 1, points.sortedBy { it.first })
    }

    private fun build(node: Int, l: Int, r: Int, points: List<Pair<Int, Int>>) {
        if (l == r) {
            val x = xs[l]
            tree[node] = points.filter { it.first == x }.map { it.second }.sorted()
            return
        }
        val mid = (l + r) / 2
        build(node * 2, l, mid, points)
        build(node * 2 + 1, mid + 1, r, points)
        tree[node] = merge(tree[node * 2], tree[node * 2 + 1])
    }

    private fun merge(a: List<Int>, b: List<Int>): List<Int> {
        val merged = mutableListOf<Int>()
        var i = 0
        var j = 0
        while (i < a.size && j < b.size) {
            if (a[i] <= b[j]) merged.add(a[i++]) else merged.add(b[j++])
        }
        while (i < a.size) merged.add(a[i++])
        while (j < b.size) merged.add(b[j++])
        return merged
    }

    fun countInRectangle(x1: Int, x2: Int, y1: Int, y2: Int): Int {
        return countQuery(1, 0, xs.size - 1, x1, x2, y1, y2)
    }

    fun existsInRectangle(x1: Int, x2: Int, y1: Int, y2: Int): Boolean {
        return existsQuery(1, 0, xs.size - 1, x1, x2, y1, y2)
    }

    private fun countQuery(node: Int, l: Int, r: Int, x1: Int, x2: Int, y1: Int, y2: Int): Int {
        if (xs[r] < x1 || xs[l] > x2) return 0
        if (x1 <= xs[l] && xs[r] <= x2) {
            val list = tree[node]
            val leftIdx = lowerBound(list, y1)
            val rightIdx = upperBound(list, y2)
            return rightIdx - leftIdx
        }
        val mid = (l + r) / 2
        return countQuery(node * 2, l, mid, x1, x2, y1, y2) +
                countQuery(node * 2 + 1, mid + 1, r, x1, x2, y1, y2)
    }

    private fun existsQuery(node: Int, l: Int, r: Int, x1: Int, x2: Int, y1: Int, y2: Int): Boolean {
        if (xs[r] < x1 || xs[l] > x2) return false
        if (x1 <= xs[l] && xs[r] <= x2) {
            val list = tree[node]
            val leftIdx = lowerBound(list, y1)
            return leftIdx < list.size && list[leftIdx] <= y2
        }
        val mid = (l + r) / 2
        return existsQuery(node * 2, l, mid, x1, x2, y1, y2) ||
                existsQuery(node * 2 + 1, mid + 1, r, x1, x2, y1, y2)
    }

    private fun lowerBound(list: List<Int>, value: Int): Int {
        var low = 0
        var high = list.size
        while (low < high) {
            val mid = (low + high) / 2
            if (list[mid] < value) low = mid + 1 else high = mid
        }
        return low
    }

    private fun upperBound(list: List<Int>, value: Int): Int {
        var low = 0
        var high = list.size
        while (low < high) {
            val mid = (low + high) / 2
            if (list[mid] <= value) low = mid + 1 else high = mid
        }
        return low
    }
}


fun maxRectangleArea(xCoord: IntArray, yCoord: IntArray): Long {
    val n = xCoord.size
    val tree = SegmentTree2D(List(n) { xCoord[it] to yCoord[it] })
    val xy = mutableMapOf<Int, MutableList<Int>>()
    val yx = mutableMapOf<Int, TreeSet<Int>>()

    for (i in 0 until n) {
        val x = xCoord[i]
        val y = yCoord[i]
        xy.computeIfAbsent(x) { mutableListOf() }.add(y)
        yx.computeIfAbsent(y) { TreeSet() }.add(x)
    }

    var maxArea = -1L
    for ((startX, yList) in xy) {
        yList.sort()
        for (i in 0 until yList.size - 1) {
            val y1 = yList[i]
            val y2 = yList[i + 1]
            val endX1 = yx[y1]?.higher(startX) ?: continue
            val endX2 = yx[y2]?.higher(startX) ?: continue
            if (endX1 != endX2) continue
            if (tree.existsInRectangle(startX, endX2, y1, y2)) continue
            val area = (endX2 - startX).toLong() * (y2 - y1).toLong()
            //   println("Area = $startX $endX2 $y1 $y2 = $area")
            maxArea = maxOf(maxArea, area)
        }
    }
    return maxArea

}

fun maxAbsValExpr(arr1: IntArray, arr2: IntArray): Int {
    val n = arr1.size
    val mins = IntArray(4) { Int.MAX_VALUE }
    val maxes = IntArray(4) { Int.MIN_VALUE }
    var result = Int.MIN_VALUE
    for (i in 0 until n) {
        val a = arr1[i]
        val b = arr2[i]
        mins[0] = minOf(mins[0], a + b + i)
        mins[1] = minOf(mins[1], a - b + i)
        mins[2] = minOf(mins[2], -a + b + i)
        mins[3] = minOf(mins[3], -a - b + i)

        maxes[0] = maxOf(maxes[0], a + b + i)
        maxes[1] = maxOf(maxes[1], a - b + i)
        maxes[2] = maxOf(maxes[2], -a + b + i)
        maxes[3] = maxOf(maxes[3], -a - b + i)
    }
    for (i in 0 until 4) {
        result = maxOf(result, maxes[i] - mins[i])
    }
    return result
}

fun mergeKLists(lists: Array<ListNode?>): ListNode? {
    val root = ListNode(0)
    var result: ListNode? = root
    while (true) {
        var minVal = Int.MAX_VALUE
        var minIndex = -1
        for (i in lists.indices) {
            val value = lists[i]?.`val` ?: continue
            if (value < minVal) {
                minIndex = i
                minVal = value
            }
        }
        if (minIndex < 0) break
        val newNode = ListNode(minVal)
        result?.next = newNode
        result = result?.next
        lists[minIndex] = lists[minIndex]?.next
    }
    result?.next = null
    return root.next
}

fun frequencySort(s: String): String {
    val char2Freq = mutableMapOf<Char, Int>()
    for (c in s) {
        val f = char2Freq[c] ?: 0
        char2Freq[c] = f + 1
    }
    return char2Freq.toList()
        .sortedByDescending { it.second }
        .joinToString("") { (c, f) ->
            val builder = StringBuilder()
            for (i in 0 until f) builder.append(c)
            builder.toString()
        }
}

fun findClosest(x: Int, y: Int, z: Int): Int {
    val first = abs(x - z)
    val second = abs(y - z)
    return when {
        first == second -> 0
        first > second -> 2
        else -> 1
    }
}

fun longestPalindrome(words: Array<String>): Int {
    var length = 0
    val freq = mutableMapOf<String, Int>()

    for (word in words) {
        val reversed = word.reversed()
        val f = freq[reversed] ?: 0
        if (f > 0) {
            freq[reversed] = f - 1
            length += 4
            continue
        }
        freq[word] = (freq[word] ?: 0) + 1
    }
    val hasOddPalindrome = freq.any {
        it.value % 2 == 1 && it.key[0] == it.key[1]
    }
    if (hasOddPalindrome) length += 2
    return length
}


fun palindromePairs(words: Array<String>): List<List<Int>> {
    val n = words.size
    val base = 131L
    val mod = 1_000_000_007L

    val maxLen = words.maxOf { it.length }
    val pow = LongArray(maxLen + 1)
    pow[0] = 1
    for (i in 1..maxLen) pow[i] = (pow[i - 1] * base) % mod

    fun hashForward(s: String): Long {
        var h = 0L
        for (c in s) h = (h * base + c.code) % mod
        return h
    }

    fun hashBackward(s: String): Long {
        var h = 0L
        for (i in s.indices.reversed()) h = (h * base + s[i].code) % mod
        return h
    }

    val fHash = LongArray(n) { hashForward(words[it]) }
    val bHash = LongArray(n) { hashBackward(words[it]) }
    val len = IntArray(n) { words[it].length }

    val wordMap = HashMap<Long, Int>()
    for (i in 0 until n) {
        wordMap[fHash[i]] = i
    }

    fun isPalindromePair(i: Int, j: Int): Boolean {
        val left = (fHash[i] * pow[len[j]] + fHash[j]) % mod
        val right = (bHash[j] * pow[len[i]] + bHash[i]) % mod
        return left == right
    }

    val result = mutableSetOf<List<Int>>()

    for (i in 0 until n) {
        val w = words[i]
        val m = w.length

        var revPrefHash = 0L
        for ((j, c) in w.withIndex()) {
            // prefix [0..j]
            revPrefHash = (revPrefHash + c.code * pow[j]) % mod
            val idx = wordMap[revPrefHash]
            if (idx != null && idx != i && isPalindromePair(i, idx)) {
                result.add(listOf(i, idx)) // i + idx
            }
        }

        var revSuffHash = 0L
        for (j in m - 1 downTo 0) {
            // suffix [j..m-1]
            revSuffHash = (revSuffHash * base + w[j].code) % mod
            val idx = wordMap[revSuffHash]
            if (idx != null && idx != i && isPalindromePair(idx, i)) {
                result.add(listOf(idx, i)) // idx + i
            }
        }
    }

    val emptyIndex = words.indexOf("")
    if (emptyIndex != -1) {
        for (i in words.indices) {
            if (i == emptyIndex) continue
            if (fHash[i] != bHash[i]) continue
            result.add(listOf(emptyIndex, i))
            result.add(listOf(i, emptyIndex))
        }
    }

    return result.toList()
}

fun makeTheIntegerZero(num1: Int, num2: Int): Int {
    val a = num1.toLong()
    val b = num2.toLong()


    for (k in 1..1000) {
        val num = a - b * k
        if (num <= 0) continue
        //  println("$num : ${num.toString(2)}")
        val bitCount = num.countOneBits()
        if (num == 1L && k > 1) continue
        if (bitCount <= k) return k
    }
    return -1
}

fun brokenCalc(startValue: Int, target: Int): Int {
    var steps = 0
    var x = target

    while (x > startValue) {
        if (x and 1 == 0) {
            x = x shr 1
        } else {
            x++
        }
        steps++
    }

    return steps + (startValue - x)
}

fun minOperations(queries: Array<IntArray>): Long {

    fun prefix(n: Long): Long {
        if (n <= 0) return 0
        var k = 0
        var pow = 1L
        while (pow * 4 <= n) {
            pow *= 4
            k++
        }
        val gk = (1 - (k + 1) * pow + k * pow * 4) / 3
        return gk + (k + 1).toLong() * (n - pow + 1)
    }

    var result = 0L
    for ((start, end) in queries) {
        val cnt = prefix(end.toLong()) - prefix(start.toLong() - 1L)
        result += ceil(0.5 * cnt).toLong()
    }
    return result
}

fun countPrefixSuffixPairs(words: Array<String>): Long {

    val n = words.maxOf { it.length }

    val mod1 = 1_000_000_007L
    val base1 = 131L
    val pow1 = LongArray(n + 1)

    pow1[0] = 1L

    for (i in 0 until n) {
        pow1[i + 1] = (pow1[i] * base1) % mod1
    }

    fun createHash(word: String): Long {
        var hash1 = 0L
        for (c in word) {
            hash1 = (hash1 * base1 + c.code) % mod1
        }
        return hash1
    }

    val hashMap = mutableMapOf<Long, Int>()

    // println(hashMap)

    class Node {
        val children = mutableMapOf<Char, Node>()
        var words: MutableSet<Long>? = null
    }

    val forward = Node()
    val backward = Node()
    var cnt = 0L

    for (i in words.indices) {
        val word = words[i]
        val hash = createHash(word)

        var fNode = forward
        var bNode = backward

        for (i in 0 until word.length) {
            val j = word.length - 1 - i
            val front = word[i]
            val back = word[j]

            val nextFront = fNode.children[front]
            val nextBack = bNode.children[back]
            if (nextFront == null || nextBack == null) break
            val frontSet = nextFront.words
            val backSet = nextBack.words
            val intersectCount = if (frontSet.isNullOrEmpty() || backSet.isNullOrEmpty()) {
                0L
            } else {
                frontSet.intersect(backSet).sumOf { hashMap[it] ?: 0 }.toLong()
            }


            cnt += intersectCount
            fNode = nextFront
            bNode = nextBack
        }

        fNode = forward
        for (j in word.indices) {
            val c = word[j]
            fNode = fNode.children.computeIfAbsent(c) { Node() }
        }
        if (fNode.words.isNullOrEmpty()) {
            fNode.words = mutableSetOf(hash)
        } else {
            fNode.words?.add(hash)
        }

        bNode = backward
        for (j in word.length - 1 downTo 0) {
            val c = word[j]
            bNode = bNode.children.computeIfAbsent(c) { Node() }
        }
        if (bNode.words.isNullOrEmpty()) {
            bNode.words = mutableSetOf(hash)
        } else {
            bNode.words?.add(hash)
        }

        hashMap[hash] = (hashMap[hash] ?: 0) + 1
    }

    return cnt
}

fun peopleAwareOfSecret(n: Int, delay: Int, forget: Int): Int {
    val mod = 1_000_000_007L
    val dp = LongArray(n + 1)
    dp[1] = 1

    for (day in 2..n) {
        var sum = 0L
        for (j in 1 until day) {
            if (j + delay <= day && day < j + forget) {
                sum = (sum + dp[j]) % mod
            }
        }
        dp[day] = sum
    }

    var ans = 0L
    for (i in n - forget + 1..n) {
        if (i >= 1) ans = (ans + dp[i]) % mod
    }
    return ans.toInt()
}

fun minimumTeachings(n: Int, languages: Array<IntArray>, friendships: Array<IntArray>): Int {
    val m = languages.size

    val userLanguages = mutableListOf<Set<Int>>(emptySet())
    for (list in languages) userLanguages.add(list.toSet())

    val notConnected = friendships.filter { (u, v) ->
        userLanguages[u].intersect(userLanguages[v]).isEmpty()
    }
    println(notConnected.map { it.toList() })
    if (notConnected.isEmpty()) return 0

    var minTeachings = m
    for (i in 1..n) {
        val teachSet = mutableSetOf<Int>()

        for ((u, v) in notConnected) {
            if (i !in userLanguages[u]) teachSet.add(u)
            if (i !in userLanguages[v]) teachSet.add(v)
        }
        minTeachings = minOf(minTeachings, teachSet.size)
    }
    return minTeachings
}

fun main() {

    println(
        minimumTeachings(
            3,
            "[[2],[1,3],[1,2],[3]]".to2DIntArray(),
            "[[1,4],[1,2],[3,4],[2,3]]".to2DIntArray()
        )
    )
}