package topic

import java.util.PriorityQueue

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