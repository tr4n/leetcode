package topic

import java.util.*
import kotlin.math.pow

fun numSmallerByFrequency(queries: Array<String>, words: Array<String>): IntArray {
    val m = queries.size
    val n = words.size

    fun f(str: String): Int {
        var minChar = 'z' + 1
        var freq = 0

        for (c in str) {
            if (c < minChar) {
                freq = 1
                minChar = c
            } else if (c == minChar) freq++
        }
        return freq
    }

    val wordValues = words.map { f(it) }.sorted()
    val ans = IntArray(queries.size)

    for (i in 0 until m) {
        val target = f(queries[i])
        var l = 0
        var r = n - 1
        var last = -1

        while (l <= r) {
            val mid = (l + r) / 2
            val value = wordValues[mid]
            if (value <= target) {
                last = mid
                l = mid + 1
            } else r = mid - 1
        }
        ans[i] = n - last - 1
    }
    return ans
}

fun suggestedProducts(products: Array<String>, searchWord: String): List<List<String>> {
    class Node {
        val children = mutableMapOf<Char, Node>()
        var words: PriorityQueue<String>? = null
    }

    val root = Node()

    for (i in 0 until products.size) {
        val product = products[i]
        var node = root

        for (c in product) {
            node = node.children.computeIfAbsent(c) { Node() }
            if (node.words == null) {
                node.words = PriorityQueue<String>(compareByDescending { it })
            }
            node.words?.add(product)
            if ((node.words?.size ?: 0) > 3) node.words?.poll()
        }
    }
    val ans = MutableList(searchWord.length) { listOf<String>() }
    var node = root
    for (i in 0 until searchWord.length) {
        val c = searchWord[i]
        node = node.children[c] ?: break
        ans[i] = node.words?.sorted() ?: emptyList()
    }
    return ans
}

fun reorganizeString(s: String): String {
    val n = s.length
    val groups = s.groupingBy { it }.eachCount()
    val heap = PriorityQueue<Pair<Char, Int>>(compareByDescending { it.second })
    for (entry in groups) {
        heap.add(entry.key to entry.value)
    }

    val builder = StringBuilder()
    while (heap.size >= 2) {
        val (char1, cnt1) = heap.poll()
        val (char2, cnt2) = heap.poll()
        val last = builder.lastOrNull()
        if (last == char1) {
            builder.append(char2)
            builder.append(char1)
        } else {
            builder.append(char1)
            builder.append(char2)
        }
        if (cnt1 > 1) heap.add(char1 to cnt1 - 1)
        if (cnt2 > 1) heap.add(char2 to cnt2 - 1)
    }
    if (heap.isEmpty()) return builder.toString()
    val (char, cnt) = heap.poll()
    if (cnt > 1) return ""
    val first = builder.firstOrNull()
    val last = builder.lastOrNull()
    if (char == first && char == last) return ""
    if (char != last) return builder.append(char).toString()
    return builder.reversed().toString() + char
}

fun numMatchingSubseq(s: String, words: Array<String>): Int {
    val positions = Array(26) { mutableListOf<Int>() }
    for (i in 0 until s.length) {
        val c = s[i] - 'a'
        positions[c].add(i)
    }

    var cnt = 0
    for (word in words) {
        println(word)
        var last = -1
        var isSub = true
        for (c in word) {
            val list = positions[c - 'a']
            var l = 0
            var r = list.size - 1
            var pos = -1

            while (l <= r) {
                val mid = (l + r) / 2
                val value = list[mid]
                if (value >= last + 1) {
                    pos = mid
                    r = mid - 1
                } else l = mid + 1
            }
            // println("- $c: $pos")
            if (pos < 0) {
                isSub = false
                break
            }
            last = list[pos]
        }
        if (isSub) cnt++
    }
    return cnt
}

fun numFriendRequests(ages: IntArray): Int {
    val n = ages.max()
    val counts = IntArray(n + 1)
    for (age in ages) counts[age]++
    var ans = 0
    val prefix = IntArray(n + 2)
    for (i in 1..n) prefix[i + 1] = prefix[i] + counts[i]
    for (age in 1..n) {
        val xCount = counts[age]
        if (age <= 14 || xCount == 0) continue
        if (xCount >= 2) ans += xCount * (xCount - 1)
        val start = age / 2 + 8
        val yCount = prefix[age] - prefix[start]
        ans += (xCount * yCount)
    }
    return ans
}

fun sumSubseqWidths(nums: IntArray): Int {
    nums.sort()
    val mod = 1_000_000_007
    val n = nums.size
    val pow = LongArray(n + 1)
    pow[0] = 1L
    for (i in 1..n) pow[i] = (pow[i - 1] * 2) % mod
    var ans = 0L
    for (i in 0 until n) {
        val num = nums[i].toLong()
        val totalMax = num * pow[i] % mod
        val totalMin = num * pow[n - i - 1] % mod
        ans = (ans + totalMax - totalMin + mod) % mod
    }
    return ans.toInt()
}

fun maxJumps(arr: IntArray, d: Int): Int {
    val n = arr.size
    val indices = arr.indices.sortedByDescending { arr[it] }
    val dp = IntArray(n) { 1 }
    for (i in indices) {
        val num = arr[i]
        val start = (i - d).coerceAtLeast(0)
        val end = (i + d).coerceAtMost(n - 1)
        for (j in (i + 1)..end) {
            if (arr[j] >= num) break
            dp[j] = maxOf(dp[j], dp[i] + 1)
        }
        for (j in (i - 1) downTo start) {
            if (arr[j] >= num) break
            dp[j] = maxOf(dp[j], dp[i] + 1)
        }
    }


    return dp.max()
}

fun rankTeams(votes: Array<String>): String {
    val n = votes[0].length
    val teams = MutableList(26) { IntArray(n) }
    for (vote in votes) {
        for (i in 0 until n) {
            val id = vote[i] - 'A'
            teams[id][i]++
        }
    }

    return votes[0].toList().sortedWith { charA, charB ->
        val teamA = teams[charA - 'A']
        val teamB = teams[charB - 'A']
        for (i in 0 until n) {
            if (teamA[i] != teamB[i]) {
                return@sortedWith teamB[i].compareTo(teamA[i])
            }
        }
        charB.compareTo(charA)
    }.joinToString("")
}

fun getKth(lo: Int, hi: Int, k: Int): Int {
    if (hi <= lo) return lo
    val memo = mutableMapOf<Int, Int>()

    fun f(num: Int): Int {
        if (num == 1) return 0
        val value = memo[num]
        if (value != null) return value
        val ans = 1 + if (num % 2 == 0) f(num / 2) else f(3 * num + 1)
        memo[num] = ans
        return ans
    }

    val heap =
        PriorityQueue<Pair<Int, Int>>(compareByDescending<Pair<Int, Int>> { it.second }.thenByDescending { it.first })

    for (num in lo..hi) {
        val value = f(num)
        heap.add(num to value)
        if (heap.size > k) heap.poll()
    }
    println(heap)
    // var ans = lo
    // for(i in 0 until k) ans = heap.poll().first
    return heap.poll().first
}

fun checkIfCanBreak(s1: String, s2: String): Boolean {
    val x = s1.toCharArray()
    val y = s2.toCharArray()
    x.sort()
    y.sort()
    //   println(x.toList())
    //  println(y.toList())
    val n = x.size
    return (0 until n).all { x[it] <= y[it] }
            || (0 until n).all { x[it] >= y[it] }

}

fun arrangeWords(text: String): String {
    return text.split(" ").withIndex()
        .sortedWith(compareBy<IndexedValue<String>> { it.value.length }.thenBy { it.index })
        .joinToString(" ") { entry ->
            entry.value.replaceFirstChar { it.lowercase() }
        }
        .replaceFirstChar { it.uppercase() }
}

fun maximumEnergy(energy: IntArray, k: Int): Int {
    val n = energy.size
    val suffix = IntArray(k)
    val maxSuffix = IntArray(k) { Int.MIN_VALUE }
    for (i in (n - 1) downTo 0) {
        val id = (n - 1 - i) % k
        suffix[id] += energy[i]
        maxSuffix[id] = maxOf(maxSuffix[id], suffix[id])
    }
    return maxSuffix.max()
}

fun largestMultipleOfThree(digits: IntArray): String {
    val n = digits.size
    //  var dp = IntArray(3) { Int.MIN_VALUE }
    //  dp[0] = 0
    digits.sortDescending()
    var rem = 0
    val buckets = Array(3) { mutableListOf<Int>() }
    for (digit in digits) {
        buckets[digit % 3].add(digit)
        rem = (rem + digit) % 3
    }
    if (rem == 0) {
        return digits.joinToString("").trimStart('0').ifEmpty { "0" }
    }

//    var lastIndex = IntArray(3) { -1 }
//
//
//    for (i in 0 until digits.size) {
//        val digit = digits[i]
//        val nextDp = dp.clone()
//        val nextLast = lastIndex.clone()
//
//        for (r in 0 until 3) {
//            if (dp[r] == Int.MIN_VALUE) continue
//            val newR = (r + digit) % 3
//            val newLen = dp[r] + 1
//            if (newLen > nextDp[newR]) {
//                nextDp[newR] = newLen
//                nextLast[newR] = i
//            }
//        }
//
//        dp = nextDp
//        lastIndex = nextLast
//    }
//    val maxLen = dp[0]
//    if (maxLen == 0) {
//        return if (digits.any { it == 0 }) "0" else ""
//    }
//    println("maxLen : ${dp[0]}")

    // println(buckets.joinToString("\n"))
    println(rem)
    while (rem != 0) {
        when {
            buckets[rem].isNotEmpty() -> {
                buckets[rem].removeLast()
                rem = 0
            }

            buckets[3 - rem].isNotEmpty() -> {
                buckets[3 - rem].removeLast()
                rem = 3 - rem
            }

            else -> return ""
        }
    }

    val result = buckets.flatMap { it }.sortedDescending()
    if (result.isEmpty()) return ""
    return result.joinToString("").trimStart('0').ifEmpty { "0" }
}

fun maxSatisfaction(satisfaction: IntArray): Int {
    val n = satisfaction.size
    val suffix = IntArray(n + 1)
    var totalSum = 0
    var remove = 0
    for (i in (n - 1) downTo 0) {
        suffix[i] = suffix[i + 1] + satisfaction[i]
        val time = satisfaction[i] * (i + 1)
        totalSum += time
        val x = time + suffix[i + 1]
        if (x < 0) remove += x
    }
    return totalSum - remove
}

fun smallestGoodBase(input: String): String {
    val n = input.toLong()
    var ans = n - 1

    fun geometricSum(x: Long, m: Int, limit: Long): Long {
        var res = 1L
        var cur = 1L
        for (i in 1..m) {
            if (cur > limit / x) return limit + 1
            cur *= x
            if (res > limit - cur) return limit + 1
            res += cur
        }
        return res
    }


    for (m in 63 downTo 2) {
        var lo = 2L
        var hi = n.toDouble().pow(1.0 / m).toLong()
        if (hi < 2) continue

        while (lo <= hi) {
            val mid = (lo + hi) / 2
            val sum = geometricSum(mid, m, n)

            if (sum >= n) {
                if (sum == n) return mid.toString()
                hi = mid - 1
            } else lo = mid + 1
        }
    }

    return ans.toString()
}

fun advantageCount(nums1: IntArray, nums2: IntArray): IntArray {
    val list1 = nums1.withIndex().sortedBy { it.value }
    val list2 = nums2.withIndex().sortedBy { it.value }
    val n = nums1.size

    var j = 0
    val result = IntArray(n) { -1 }
    val remaining = mutableListOf<Int>()
    for (i in 0 until n) {
        val item = list2[i]
        while (j < n && list1[j].value <= item.value) {
            remaining.add(list1[j++].value)
        }
        if (j == n) break
        result[item.index] = list1[j++].value
    }

    for (i in 0 until n) {
        if (result[i] == -1) {
            result[i] = remaining.removeLastOrNull() ?: -1
        }
    }
    return result
}

fun maxPerformance(n: Int, speeds: IntArray, efficiencies: IntArray, k: Int): Int {
    val mod = 1_000_000_007L
    var sum = 0L
    var ans = 0L
    val list = speeds.zip(efficiencies).sortedByDescending { it.second }
    val pq = PriorityQueue<Int>()

    for ((speed, efficiency) in list) {
        pq.add(speed)
        sum += speed

        if (pq.size > k) sum -= pq.poll()
        ans = maxOf(ans, sum * efficiency)
    }

    return (ans % mod).toInt()
}

fun minTime(n: Int, edges: Array<IntArray>, hasApple: List<Boolean>): Int {
    val graph = Array(n) { mutableListOf<Int>() }

    for ((u, v) in edges) {
        graph[u].add(v)
        graph[v].add(u)
    }

    val visited = BooleanArray(n)

    var total = 0
    fun dfs(u: Int): Boolean {
        visited[u] = true
        var apple = hasApple[u]
        for (v in graph[u]) {
            if (visited[v]) continue
            val childHasApple = dfs(v)
            if (childHasApple) {
                total += 2
                apple = true
            }

        }
        return apple
    }
    dfs(0)
    return total
}

fun baseNeg2(n: Int): String {
    val builder = StringBuilder()
    var num = n

    while (num != 0) {
        if (num % 2 == 0) {
            builder.append(0)
        } else {
            builder.append(1)
            num--
        }
        num /= -2
    }
    return builder.toString().reversed()
}

fun hasIncreasingSubarrays(nums: List<Int>, k: Int): Boolean {
    val n = nums.size
    var i = 0
    while (i < n - 2 * k + 1) {
        var incA = true
        for (a in i + 1 until i + k) {
            if (nums[a] <= nums[a - 1]) {
                incA = false
                break
            }
        }
        if (!incA) {
            i++
            continue
        }

        var incB = true
        for (b in i + k + 1 until i + 2 * k) {
            if (nums[b] <= nums[b - 1]) {
                incB = false
                break
            }
        }
        if (!incB) {
            i++
            continue
        }
        return true
    }
    return false
}

fun maxIncreasingSubarrays(nums: List<Int>): Int {
    val n = nums.size
    val inc = IntArray(n) { 1 }
    for (i in 1 until n) if (nums[i] > nums[i - 1]) inc[i] = inc[i - 1] + 1

    var lo = 1
    var hi = n / 2
    var ans = 1
    while (lo <= hi) {
        val k = (lo + hi) / 2
        var found = false
        for (i in k - 1 until n - k) {
            if (inc[i] >= k && inc[i + k] >= k) {
                found = true
                break
            }
        }

        if (found) {
            ans = k
            lo = k + 1
        } else hi = k - 1
    }
    return ans
}

fun sumOfGoodSubsequences(nums: IntArray): Int {
    val n = nums.size
    val mod = 1_000_000_007L
    val count = mutableMapOf<Long, Long>()
    val prefixSum = mutableMapOf<Long, Long>()
    var ans = 0L

    for (i in 0 until n) {
        val num = nums[i].toLong()
        var sum = num
        var cnt = 1L

        for (prev in listOf(num - 1, num + 1)) {
            val prevCount = count[prev] ?: 0L
            val prevSum = prefixSum[prev] ?: 0L
            cnt = (cnt + prevCount) % mod
            sum = (sum + prevSum + prevCount * num % mod) % mod
        }
        count[num] = (count[num] ?: 0) + cnt
        prefixSum[num] = (prefixSum[num] ?: 0) + sum

        ans = (ans + sum) % mod
    }
    return ans.toInt()
}

fun minZeroArray2(nums: IntArray, queries: Array<IntArray>): Int {
    val n = nums.size
    if (nums.all { it == 0 }) return 0

    var lo = 1
    var hi = queries.size
    var ans = -1

    while (lo <= hi) {
        val k = (lo + hi) / 2
        val delta = IntArray(n + 1)

        for (q in 0 until k) {
            val (l, r, v) = queries[q]
            delta[l] += v
            delta[r + 1] -= v
        }

        val levels = IntArray(n)
        levels[0] = delta[0]
        for (i in 1 until n) {
            levels[i] = levels[i - 1] + delta[i]
        }

        var isZero = true
        for (i in 0 until n) {
            if (nums[i] > levels[i]) {
                isZero = false
                break
            }
        }
        if (isZero) {
            ans = k
            hi = k - 1
        } else lo = k + 1
    }

    return ans
}

fun maxRemoval(nums: IntArray, queries: Array<IntArray>): Int {
    val n = nums.size
    if (nums.all { it == 0 }) return 0

    queries.sortWith(compareBy<IntArray> { it[0] }.thenByDescending { it[1] })

    var lo = 1
    var hi = queries.size
    var ans = -1

    while (lo <= hi) {
        val k = (lo + hi) / 2
        val delta = IntArray(n + 1)

        for (q in 0 until k) {
            val (l, r, v) = queries[q]
            delta[l] += v
            delta[r + 1] -= v
        }

        var acc = 0
        var isZero = true
        for (i in 0 until n) {
            acc += delta[i]
            if (nums[i] > acc) {
                isZero = false
                break
            }
        }

        if (isZero) {
            ans = k
            hi = k - 1
        } else lo = k + 1
    }

    return ans
}

fun minZeroArray(nums: IntArray, queries: Array<IntArray>): Int {
    val n = nums.size
    val m = queries.size
    if (nums.all { it == 0 }) return 0

    val dp = Array(n) { BooleanArray(nums[it] + 1) }
    for (i in 0 until n) dp[i][0] = true

    fun applyQuery(query: IntArray) {
        val (l, r, v) = query
        for (i in l..r) {
            val target = nums[i]
            if (target == 0) continue

            for (j in target downTo v) {
                dp[i][j] = dp[i][j] or dp[i][j - v]
            }
        }
    }

    for (k in 0 until m) {
        applyQuery(queries[k])
        if (dp.all { it.last() }) return k + 1
    }
    return -1
}

fun main() {
    println(
        sumOfGoodSubsequences(intArrayOf(10, 10, 1, 9))
    )
}