package topic

class LongBasisXor(private val maxBits: Int = Long.SIZE_BITS) {
    private val basis = LongArray(maxBits)

    fun insert(num: Long) {
        var x = num
        for (i in (maxBits - 1) downTo 0) {
            if ((x and (1L shl i)) == 0L) continue
            if (basis[i] == 0L) {
                basis[i] = x
                return
            }
            x = x xor basis[i]
        }
    }

    fun insert(nums: List<Long>) {
        for (num in nums) {
            var x = num
            for (i in (maxBits - 1) downTo 0) {
                if ((x and (1L shl i)) == 0L) continue
                if (basis[i] == 0L) {
                    basis[i] = x
                    break
                }
                x = x xor basis[i]
            }
        }
    }

    fun getMaxXor(): Long {
        var res = 0L
        for (i in (maxBits - 1) downTo 0) {
            res = maxOf(res, res xor basis[i])
        }
        return res
    }

    fun canRepresent(num: Long): Boolean {
        var x = num
        for (i in (maxBits - 1) downTo 0) {
            if (x and (1L shl i) == 0L) continue
            if (basis[i] == 0L) return false
            x = x xor basis[i]
        }
        return true
    }

}

fun findMaximumXOR(nums: IntArray): Int {
    class Node {
        var left: Node? = null
        var right: Node? = null
    }

    val root = Node()

    for (num in nums) {
        var node = root
        for (i in 31 downTo 0) {
            val bit = (num shr i) and 1
            if (bit == 0) {
                if (node.left == null) node.left = Node()
                node = node.left!!
            } else {
                if (node.right == null) node.right = Node()
                node = node.right!!
            }
        }
    }

    fun query(num: Int): Int {
        var node = root
        var result = 0
        for (i in 31 downTo 0) {
            val bit = (num shr i) and 1
            val left = node.left
            val right = node.right
            when {
                bit == 0 && right != null -> {
                    result = result or (1 shl i)
                    node = right
                }

                bit == 0 && left != null -> node = left

                bit == 1 && left != null -> {
                    result = result or (1 shl i)
                    node = left
                }

                bit == 1 && right != null -> node = right
                else -> break
            }
        }
        return result
    }

    return nums.maxOf { query(it) }
}

fun maximumStrongPairXor(nums: IntArray): Int {
    nums.sort()
    val n = nums.size

    class Node {
        var left: Node? = null
        var right: Node? = null
        var count = 0
    }

    val root = Node()

    fun insert(num: Int) {
        var node = root
        for (i in 31 downTo 0) {
            val bit = (num shr i) and 1
            if (bit == 0) {
                if (node.left == null) node.left = Node()
                node = node.left!!
            } else {
                if (node.right == null) node.right = Node()
                node = node.right!!
            }
            node.count++
        }
    }

    fun remove(num: Int) {
        var node = root
        for (i in 31 downTo 0) {
            val bit = (num shr i) and 1
            node = if (bit == 0) {
                node.left ?: break
            } else {
                node.right ?: break
            }
            node.count--
        }
    }

    fun query(num: Int): Int {
        var result = 0
        var node = root
        for (i in 31 downTo 0) {
            val bit = (num shr i) and 1
            val left = node.left?.takeIf { it.count > 0 }
            val right = node.right?.takeIf { it.count > 0 }
            when {
                bit == 0 && right != null -> {
                    result = result or (1 shl i)
                    node = right
                }

                bit == 0 && left != null -> node = left
                bit == 1 && left != null -> {
                    result = result or (1 shl i)
                    node = left
                }

                bit == 1 && right != null -> node = right
                else -> break
            }
        }
        return result
    }

    var ans = 0
    var j = 0
    for (x in nums) {
        while (j < n && nums[j] <= 2 * x) {
            insert(nums[j])
            j++
        }
        ans = maxOf(ans, query(x))
        remove(x)
    }
    return ans
}

fun maximizeXor(nums: IntArray, queries: Array<IntArray>): IntArray {
    class Node {
        var left: Node? = null
        var right: Node? = null
    }

    val root = Node()
    fun insert(num: Int) {
        var node = root
        for (i in 31 downTo 0) {
            val bit = (num shr i) and 1
            if (bit == 0) {
                if (node.left == null) node.left = Node()
                node = node.left!!
            } else {
                if (node.right == null) node.right = Node()
                node = node.right!!
            }
        }
    }

    fun queryMaxXorWith(num: Int): Int {
        var node = root
        var result = 0

        for (i in 31 downTo 0) {
            val bit = (num shr i) and 1
            val left = node.left
            val right = node.right

            when {
                bit == 0 && right != null -> {
                    result = result or (1 shl i)
                    node = right
                }

                bit == 0 && left != null -> node = left
                bit == 1 && left != null -> {
                    result = result or (1 shl i)
                    node = left
                }

                bit == 1 && right != null -> node = right
                else -> break
            }
        }
        return result
    }

    val n = nums.size
    nums.sort()
    val queryList = queries.withIndex().sortedBy { it.value[1] }
    val ans = IntArray(queryList.size) { -1 }
    var i = 0
    for ((index, query) in queryList) {
        val (x, m) = query
        while (i < n && nums[i] <= m) insert(nums[i++])
        if (i == 0) continue
        ans[index] = queryMaxXorWith(x)
    }
    return ans
}

fun maximumXOR(nums: IntArray): Int {
    return nums.reduce { acc, i ->
        acc or i
    }
}

fun maximumXorProduct(a: Long, b: Long, n: Int): Int {
    var x = 0L
    var xorA = a
    var xorB = b
    for (i in (n - 1) downTo 0) {
        val aBit = (a shr i) and 1L
        val bBit = (b shr i) and 1L
        if (aBit == 0L && bBit == 0L) {
            x = x or (1L shl i)
        }
    }
    val mod = 1_000_000_007L
    val finalA = xorA % mod
    val finalB = xorB % mod
    val product = (finalA * finalB) % mod
    println("x = $x, xorA = $xorA, xorB = $xorB")
    return product.toInt()
}

fun decode(encoded: IntArray): IntArray {
    val n = encoded.size + 1
    var firstNum = 0
    for (i in 0 until n) {
        firstNum = firstNum xor (i + 1)
        if (i % 2 != 0) firstNum = firstNum xor encoded[i]
    }
    val result = IntArray(n)
    result[0] = firstNum
    for (i in 1 until n) {
        result[i] = encoded[i - 1] xor result[i - 1]
    }
    return result
}

fun countPairs(nums: IntArray, low: Int, high: Int): Int {
    class Node {
        var left: Node? = null
        var right: Node? = null
    }

    val root = Node()
    fun insert(num: Int) {
        var node = root
        for (i in 31 downTo 0) {
            val bit = (num shr i) and 1
            if (bit == 0) {
                if (node.left == null) node.left = Node()
                node = node.left!!
            } else {
                if (node.right == null) node.right = Node()
                node = node.right!!
            }
        }
    }

    fun queryMaxXorWith(num: Int): Int {
        var node = root
        var result = 0

        for (i in 31 downTo 0) {
            val bit = (num shr i) and 1
            val left = node.left
            val right = node.right

            when {
                bit == 0 && right != null -> {
                    result = result or (1 shl i)
                    node = right
                }

                bit == 0 && left != null -> node = left
                bit == 1 && left != null -> {
                    result = result or (1 shl i)
                    node = left
                }

                bit == 1 && right != null -> node = right
                else -> break
            }
        }
        return result
    }

    fun queryMinXorWith(num: Int): Int {
        var node = root
        var result = 0

        for (i in 31 downTo 0) {
            val bit = (num shr i) and 1
            val left = node.left
            val right = node.right

            when {
                bit == 0 && left != null -> {
                    result = result or (1 shl i)
                    node = left
                }

                bit == 0 && right != null -> node = right
                bit == 1 && right != null -> {
                    result = result or (1 shl i)
                    node = right
                }

                bit == 1 && left != null -> node = left
                else -> break
            }
        }
        return result
    }

    val n = nums.size
    insert(nums[0])

    return 0
}

fun countTriplets(arr: IntArray): Int {
    val n = arr.size
    val prefix = IntArray(n + 1)
    for (i in 0 until n) prefix[i + 1] = prefix[i] xor arr[i]

    var cnt = 0
    for (i in 0 until n - 1) {
        for (k in i + 1 until n) {
            if (prefix[k + 1] == prefix[i]) cnt += (k - i)
        }
    }

    return cnt
}

fun findTheLongestSubstring(s: String): Int {
    val n = s.length
    val firstSeen = IntArray(64) { n }
    firstSeen[0] = -1
    var best = 0
    var status = 0
    for (last in 0 until n) {
        val mask = when (s[last]) {
            'u' -> 1 shl 1
            'e' -> 1 shl 2
            'o' -> 1 shl 3
            'a' -> 1 shl 4
            'i' -> 1 shl 5
            else -> 0
        }
        status = status xor mask

        val first = firstSeen[status]
        if (first < n) {
            best = maxOf(best, last - first)
        } else {
            firstSeen[status] = last
        }

    }
    return best
}

fun numSplits(s: String): Int {
    val n = s.length
    var set = mutableSetOf<Char>()
    val left = IntArray(n)
    val right = IntArray(n)
    for (i in 0 until n) {
        set.add(s[i])
        left[i] = set.size
    }
    set = mutableSetOf()
    for (i in (n - 1) downTo 0) {
        set.add(s[i])
        right[i] = set.size
    }

    var cnt = 0
    for (i in 0 until n - 1) {
        if (left[i] == right[i + 1]) cnt++
    }
    return cnt
}


fun longestAwesome(s: String): Int {
    val n = s.length
    val firstSeen = mutableMapOf<Int, Int>()
    firstSeen[0] = -1
    var best = 0
    var status = 0
    for (last in 0 until n) {
        val mask = 1 shl (s[last] - '0')
        status = status xor mask

        val evenFirst = firstSeen[status]
        if (evenFirst == null) {
            firstSeen[status] = last
        } else {
            best = maxOf(best, last - evenFirst)
        }

        for (c in '0'..'9') {
            val oddMask = 1 shl (c - '0')
            val oddStatus = status xor oddMask
            val oddFirst = firstSeen[oddStatus] ?: continue
            best = maxOf(best, last - oddFirst)
        }
    }
    return best
}

fun minOperations(nums: IntArray): Int {
    val totalOneBits = nums.sumOf { it.countOneBits() }
    val highestBits = nums.maxOf { 32 - it.countLeadingZeroBits() }

    return highestBits + totalOneBits - 1
}

fun stepsToZero(num: Int): Int {
    var n = num
    var steps = 0
    while (n > 0) {
        steps++
        n = if (n % 2 == 0) n / 2 else n - 1
    }
    return steps
}

fun stepsToZeroFast(num: Int): Int {
    if (num == 0) return 0
    val bits = 32 - Integer.numberOfLeadingZeros(num)
    val ones = Integer.bitCount(num)
    return bits + ones - 1
}

fun maximumBobPoints(numArrows: Int, aliceArrows: IntArray): IntArray {

    val bobArrows = IntArray(12)
    var maxPoints = 0
    var result = mutableListOf<Int>()
    fun backtrack(pos: Int, remaining: Int, score: Int) {
        if (remaining < 0) return
        val potential = score + (pos * (pos + 1)) / 2
        if (potential <= maxPoints) return

        if (pos == 0) {
            if (score > maxPoints) {
                maxPoints = score
                result = bobArrows.toMutableList()
                result[0] = remaining
            }
            return
        }


        // Bob take win this section
        val arrow = aliceArrows[pos] + 1
        if (arrow <= remaining) {
            bobArrows[pos] = arrow
            backtrack(pos - 1, remaining - arrow, score + pos)
            bobArrows[pos] = 0
        }

        // Alice take win this section
        bobArrows[pos] = 0
        backtrack(pos - 1, remaining, score)
    }

    backtrack(11, numArrows, 0)
    return result.toIntArray()
}

fun countExcellentPairs(nums: IntArray, k: Int): Long {

    val list = nums.distinct().sortedBy { it.countOneBits() }
    val n = list.size

    var ans = 0L
    for (i in 0 until n) {
        val num = list[i]
        val numCount = num.countOneBits()
        if (2L * numCount >= k) {
            ans++
        }
        var hi = i
        var lo = 0
        var first = -1
        while (lo <= hi) {
            val mid = (lo + hi) / 2
            val sum = list[mid].countOneBits() + numCount
            //    println("${list[mid]} + $num = $sum")
            if (sum < k) {
                lo = mid + 1
            } else {
                first = mid
                hi = mid - 1
            }
        }
        if (first < 0) continue
        val lessCount = 2L * (i - first)
        //   println("${list[first]}-$num: $lessCount")
        ans += lessCount
    }

    return ans
}

fun maximumRows(matrix: Array<IntArray>, numSelect: Int): Int {
    val m = matrix.size
    val n = matrix[0].size

    var maxBits = 0
    val nums = matrix.map { row ->
        var num = 0
        for (bit in row) num = (num shl 1) or bit
        maxBits = maxOf(maxBits, 32 - num.countLeadingZeroBits())
        num
    }

    val limit = (1 shl n)

    var maxRows = 0
    for (mask in 0 until limit) {
        val bitCount = mask.countOneBits()
        if (bitCount != numSelect) continue
        val iMask = mask.inv()
        val cover = nums.count { it and iMask == 0 }
        maxRows = maxOf(maxRows, cover)
    }

    return maxRows
}

fun xorAllNums(nums1: IntArray, nums2: IntArray): Int {
    val xor1 = if (nums2.size % 2 == 0) 0 else nums1.fold(0, Int::xor)
    val xor2 = if (nums1.size % 2 == 0) 0 else nums2.fold(0, Int::xor)
    return xor1 xor xor2
}

fun minimizeXor(num1: Int, num2: Int): Int {
    var bitCount = num2.countOneBits()

    var x = 0
    for (i in 31 downTo 0) {
        if (bitCount <= 0) break
        val bit = (num1 shr i) and 1
        if (bit == 1) {
            x = x or (1 shl i)
            bitCount--
        }
    }
    for (i in 0 until 31) {
        if (bitCount <= 0) break
        val bit = (num1 shr i) and 1
        if (bit == 0) {
            x = x or (1 shl i)
            bitCount--
        }
    }
    return x
}

fun findArray(pref: IntArray): IntArray {
    val n = pref.size
    val result = IntArray(n)
    var remainingXor = pref[n - 1]
    for (i in (n - 1) downTo 1) {
        result[i] = pref[i] xor pref[i - 1]
        remainingXor = remainingXor xor result[i]
    }
    result[0] = remainingXor
    return result
}

fun makeStringsEqual(s: String, target: String): Boolean {
    val n = s.length
    if (s == target) return true

    var oneBits = 0
    for (c in s) if (c == '1') oneBits++

    for (i in 0 until n) {
        val a = s[i]
        val b = target[i]
        if (a == b) continue
        if (a == '0') {
            if (oneBits <= 0) return false
            oneBits++
        }
    }

    for (i in 0 until n) {
        val a = s[i]
        val b = target[i]
        if (a == b) continue
        if (a == '1') {
            if (oneBits <= 1) return false
            oneBits--
        }
    }

    return true
}

fun substringXorQueries(s: String, queries: Array<IntArray>): Array<IntArray> {
    val seen = mutableMapOf<Int, IntArray>()
    val n = s.length
    for (i in 0 until n) {
        if (s[i] == '0') {
            if (seen[0] == null) seen[0] = intArrayOf(i, i)
            continue
        }
        var num = 0
        for (len in 0 until minOf(30, n - i)) {
            num = (num shl 1) or (s[i + len] - '0')
            if (seen[num] == null) {
                seen[num] = intArrayOf(i, i + len)
            }
        }
    }

    return Array(queries.size) {
        val target = queries[it][0] xor queries[it][1]
        seen[target] ?: intArrayOf(-1, -1)
    }
}

fun largestCombination(candidates: IntArray): Int {
    val counts = IntArray(32) {
        var cnt = 0
        for (num in candidates) {
            if ((num shr it) and 1 == 1) cnt++
        }
        cnt
    }

    return counts.max()
}


fun main() {
    println(
        largestCombination(intArrayOf(16, 17, 71, 62, 12, 24, 14))
    )
}