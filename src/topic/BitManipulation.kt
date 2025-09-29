package topic

import remote.ANDSparseTable

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
    var xorA = a
    var xorB = b
    for (i in (n - 1) downTo 0) {
        val aBit = (a shr i) and 1L
        val bBit = (b shr i) and 1L
        val mask = 1L shl i
        when {
            aBit == bBit -> {
                xorA = xorA or mask
                xorB = xorB or mask
            }

            xorA < xorB -> {
                xorA = xorA or mask
                xorB = xorB and mask.inv()
            }

            else -> {
                xorB = xorB or mask
                xorA = xorA and mask.inv()
            }
        }
    }
    val mod = 1_000_000_007L
    val finalA = xorA % mod
    val finalB = xorB % mod
    val product = (finalA * finalB) % mod
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

fun minOperations1(nums: IntArray): Int {
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


fun maximumOr(nums: IntArray, k: Int): Long {
    val n = nums.size
    val prefix = LongArray(n + 1)
    val suffix = LongArray(n + 1)

    for (i in 0 until n) {
        prefix[i + 1] = prefix[i] or nums[i].toLong()
    }
    for (i in (n - 1) downTo 0) {
        suffix[i] = suffix[i + 1] or nums[i].toLong()
    }

    var maxOr = 0L

    for (i in 0 until n) {
        val num = nums[i].toLong() shl k
        val pre = prefix[i]
        val suf = suffix[i + 1]
        val value = pre or suf or num
        maxOr = maxOf(maxOr, value)
    }

    return maxOr
}

fun minImpossibleOR(nums: IntArray): Int {
    val set = nums.toSet()
    for (i in 0 until 32) {
        val num = 1 shl i
        if (num !in set) return num
    }
    return nums.max() + 1
}

fun doesValidArrayExist(derived: IntArray): Boolean {
    return derived.fold(0, Int::xor) == 0
}

fun findThePrefixCommonArray(A: IntArray, B: IntArray): IntArray {
    val result = IntArray(A.size)
    var maskA = 0L
    var maskB = 0L

    for (i in 0 until A.size) {
        maskA = maskA or (1L shl A[i])
        maskB = maskB or (1L shl B[i])
        result[i] = (maskA and maskB).countOneBits()
    }
    return result
}

fun beautifulSubarrays(nums: IntArray): Long {
    val seen = mutableMapOf<Int, Int>()
    seen[0] = 1
    val n = nums.size
    var status = 0
    var ans = 0L
    for (i in 0 until n) {
        val num = nums[i]

        for (j in 21 downTo 0) {
            if ((num shr j) and 1 == 1) {
                status = status xor (1 shl j)
            }
        }
        val seenCount = seen[status] ?: 0
        ans += seenCount.toLong()
        seen[status] = seenCount + 1
    }

    return ans
}

fun squareFreeSubsets(nums: IntArray): Int {
    val mod = 1_000_000_007L
    val primes = intArrayOf(2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
    var cnt1 = nums.count { it == 1 }
    val counts = nums.filterNot { num ->
        num == 1 || primes.any { num % (it * it) == 0 }
    }.map { num ->
        var binary = 0
        for (i in 0 until primes.size) {
            if (num % primes[i] == 0) {
                binary = binary or (1 shl i)
            }
        }
        binary
    }.groupingBy { it }.eachCount()

    val maxMask = (1 shl (primes.size))
    val dp = LongArray(maxMask)
    dp[0] = 1L

    for ((num, cnt) in counts) {
        for (mask in (maxMask - 1) downTo 0) {
            if (dp[mask] <= 0) continue
            if (num and mask != 0) continue
            val newMask = num or mask
            dp[newMask] = (dp[newMask] + dp[mask] * cnt) % mod

        }
    }

    var ans = 0L
    for (mask in 0 until maxMask) {
        ans = (ans + dp[mask]) % mod
    }

    var pow2 = 1L
    while (cnt1-- > 0) pow2 = (pow2 * 2) % mod
    ans = (ans * pow2) % mod
    ans = (ans - 1 + mod) % mod
    return ans.toInt()
}

fun minOperations(n: Int): Int {
    var num = n
    val maxBits = 32 - n.countLeadingZeroBits()
    var cnt = 0
    for (i in 0 until maxBits + 1) {
        if ((num shr i) and 1 == 0) continue
        // println("$num ${num.toString(2)} ${num.countOneBits()}")
        val delta = 1 shl i
        val add = num + delta
        val subtract = num - delta
        num = if (add.countOneBits() <= subtract.countOneBits()) {
            add
        } else {
            subtract
        }
        cnt++
        if (num == 0) return cnt
    }
    return cnt
}

fun minOperations(nums: List<Int>, target: Int): Int {
    val maxTargetBits = 32 - target.countLeadingZeroBits()
    val maxNumBits = nums.maxOf { 32 - it.countLeadingZeroBits() }
    val maxBits = maxOf(maxTargetBits, maxNumBits)

    val bits = IntArray(maxBits + 1)
    for (num in nums) {
        val exp = Integer.numberOfTrailingZeros(num)
        bits[exp]++
    }

    //  println(target.toString(2).reversed().toList())
    //  println("---")
    //  println(bits.toList())
    var cnt = 0
    for (i in 0 until maxTargetBits) {
        val needBit = (target shr i) and 1
        val extraBits = bits[i] - needBit
        if (i + 1 < maxBits && extraBits >= 2) {
            val carry = extraBits / 2
            bits[i + 1] += carry
            bits[i] -= extraBits
        }

        if (extraBits < 0) {
            var j = i + 1
            while (j < maxBits && bits[j] == 0) j++
            if (j == maxBits) return -1
            bits[j]--
            j--
            while (j >= i) {
                bits[j--]++
                cnt++
            }
            bits[i]++
        }

        //  println(bits.toList())
    }
    return cnt
}

fun maxSubarrays2(nums: IntArray): Int {
    val n = nums.size
    val table = ANDSparseTable(nums)

    val totalAND = nums.reduce { acc, i -> acc and i }

    var l = 0
    var r = n - 1
    var pos = n - 1
    while (l <= r) {
        val mid = (l + r) / 2
        val value = table.queryAND(0, mid)
        if (value <= totalAND) {
            pos = mid
            r = mid - 1
        } else {
            l = mid + 1
        }
    }
    if (pos == n - 1) return 1
    println(nums.toList().subList(0, pos + 1))
    var cnt = 1
    pos++
    while (pos < n) {
        var l = pos
        var r = n - 1
        var index = -1
        while (l <= r) {
            val mid = (l + r) / 2
            val value = table.queryAND(pos, mid)
            if (value <= 0) {
                index = mid
                r = mid - 1
            } else {
                l = mid + 1
            }
        }
        if (index < pos) return 1
        println(nums.toList().subList(pos, index + 1))
        pos = index + 1
        cnt++
    }

    return cnt
}


fun getXORSum(arr1: IntArray, arr2: IntArray): Int {
    val allXor = arr2.fold(0, Int::xor)

    var result = 0
    for (i in 0 until arr1.size) {
        result = result xor (arr1[i] and allXor)
    }
    return result
}

fun totalHammingDistance(nums: IntArray): Int {
    val n = nums.size
    var total = 0
    for (i in 31 downTo 0) {
        var oneCount = 0
        for (num in nums) {
            if (((num shr i) and 1) == 1) oneCount++
        }
        val zeroCount = n - oneCount
        total += oneCount * zeroCount
    }
    return total
}

fun maxGoodNumber(nums: IntArray): Int {
    val (a, b, c) = nums

    fun createNum(a: Int, b: Int, c: Int): Int {
        var num = a
        num = (num shl (32 - b.countLeadingZeroBits())) or b
        num = (num shl (32 - c.countLeadingZeroBits())) or c
        return num
    }

    return maxOf(
        createNum(a, b, c),
        createNum(a, c, b),
        createNum(b, a, c),
        createNum(b, c, a),
        createNum(c, a, b),
        createNum(c, b, a)
    )

}

fun concatenatedBinary(n: Int): Int {
    val mod = 1_000_000_007
    var num = 1L
    for (i in 2..n) {
        val bits = 32 - i.countLeadingZeroBits()
        num = (num shl bits) or i.toLong()
        num %= mod
    }
    return num.toInt()
}

fun smallestSufficientTeam(req_skills: Array<String>, people: List<List<String>>): IntArray {
    val maxBits = req_skills.size
    val skillToIndex = req_skills.withIndex().associate { it.value to it.index }
    val skillPeople = people.map { skills ->
        var num = 0
        for (skill in skills) {
            val i = skillToIndex[skill] ?: 0
            num = num or (1 shl i)
        }
        num
    }
    val n = people.size
    val limit = 1 shl maxBits
    val dp = IntArray(limit) { n }
    val parent = IntArray(limit) { -1 }
    val choose = IntArray(limit) { -1 }
    dp[0] = 0

    for (p in skillPeople.withIndex()) {
        for (status in 0 until limit) {
            if (dp[status] >= n) continue
            val newStatus = status or p.value
            val newSize = dp[status] + 1
            if (newStatus == status || newSize >= dp[newStatus]) continue
            dp[newStatus] = newSize
            parent[newStatus] = status
            choose[newStatus] = p.index
        }
    }
    val ans = mutableListOf<Int>()
    var mask = limit - 1
    while (mask > 0) {
        ans.add(choose[mask])
        mask = parent[mask]
    }
    println(dp[limit - 1])
    return ans.toIntArray()
}

fun maxScoreWords(words: Array<String>, letters: CharArray, scores: IntArray): Int {

    val charCount = IntArray(26)

    for (i in letters.indices) {
        val index = letters[i] - 'a'
        charCount[index]++
    }

    val wordMap = words.map { word ->
        word.groupingBy { it - 'a' }.eachCount()
    }

    val n = words.size
    var ans = 0

    fun dfs(pos: Int, score: Int) {
        if (pos == n) {
            ans = maxOf(ans, score)
            return
        }

        dfs(pos + 1, score)

        val word = wordMap[pos]
        val hasEnoughChars = word.all { (i, cnt) ->
            charCount[i] >= cnt
        }
        if (!hasEnoughChars) return
        var newScore = score
        for ((i, cnt) in word) {
            charCount[i] -= cnt
            newScore += scores[i] * cnt
        }
        dfs(pos + 1, newScore)
        for ((i, cnt) in word) {
            charCount[i] += cnt
        }
    }

    dfs(0, 0)
    return ans
}

fun minOperations4(nums: IntArray): Int {
    val n = nums.size
    var cnt = 0
    for (i in 0 until n - 2) {
        if (nums[i] == 1) continue
        nums[i] = 1
        nums[i + 1] = 1 xor nums[i + 1]
        nums[i + 2] = 1 xor nums[i + 2]
        cnt++
    }
    return if (nums[n - 2] == 1 && nums[n - 1] == 1) cnt else -1
}

fun minEnd(n: Int, x: Int): Long {
    var num = x.toLong()
    var m = n - 1
    var i = 0
    while (m > 0) {
        if (((num shr i) and 1L) == 0L) {
            if ((m and 1) == 1) {
                num = num or (1L shl i)
            }
            m = m shr 1
        }
        i++
    }
    return num
}

fun uniqueXorTriplets(nums: IntArray): Int {
    val basis = IntArray(32)

    for (num in nums) {
        var x = num
        for (i in 31 downTo 0) {
            if (x and (1 shl i) == 0) continue
            if (basis[i] == 0) {
                basis[i] = x
                continue
            }
            x = x xor basis[i]
        }
    }

    val set = mutableSetOf<Int>()
    for (i in 0 until 30) {
        for (j in i + 1 until 31) {
            for (k in j + 1 until 32) {
                val value = basis[i] xor basis[j] xor basis[k]
                set.add(value)
            }
        }
    }
    return set.size
}

fun checkEqualPartitions(nums: IntArray, target: Long): Boolean {
    val n = nums.size
    val allProd = nums.fold(1L) { acc, item -> acc * item.toLong() }
    if (allProd != target * target) {
        return false
    }

    val limit = (1 shl n)

    for (status in 1 until limit) {
        var first = 1L
        var second = 1L
        for (j in 0 until n) {
            val bit = (status shr j) and 1
            val num = nums[j].toLong()
            if (bit == 0) first *= num else second *= num
        }
        //  println("${status.toString(2)} $first $second")
        if (first == target && second == target) return true
    }
    return false
}

fun countPalindromePaths(parent: List<Int>, s: String): Long {
    val n = parent.size

    val graph = Array(n) { mutableListOf<Int>() }
    for (i in 0 until n) {
        val p = parent[i]
        if (p != -1) graph[p].add(i)
    }

    val seen = mutableMapOf<Int, Long>()
    //  seen[0] = 1L

    var ans = 0L

    fun dfs(u: Int, status: Int) {
        val cnt = 1L + (seen[status] ?: 0L)
        seen[status] = cnt
        ans += cnt

        for (c in 0 until 26) {
            val mask = 1 shl c
            ans += seen[status xor mask] ?: 0L
        }


        for (v in graph[u]) {
            val mask = 1 shl (s[v] - 'a')
            dfs(v, status xor mask)
        }
        //  seen[status] = cnt - 1L
    }
    dfs(0, 0)
    //  println(seen)
    return ans - n
}

fun maximumRequests(n: Int, requests: Array<IntArray>): Int {
    val m = requests.size
    val buildings = IntArray(n)

    var ans = 0

    fun dfs(pos: Int, mask: Int) {
        if (pos == m) {
            if (buildings.all { it == 0 }) {
                ans = maxOf(ans, mask.countOneBits())
            }
            return
        }

        val newMask = mask or (1 shl pos)
        val from = requests[pos][0]
        val to = requests[pos][1]
        buildings[from]--
        buildings[to]++
        dfs(pos + 1, newMask)
        buildings[from]++
        buildings[to]--

        dfs(pos + 1, mask)
    }
    dfs(0, 0)
    return ans
}

fun minimumOneBitOperations(n: Int): Int {
    if (n == 0) return 0
    val k = 32 - n.countLeadingZeroBits()
    val r = n - (1 shl (k - 1))
    val sub = minimumOneBitOperations(r)
    return (1 shl k) - 1 - sub
}

fun maxStudents(seats: Array<CharArray>): Int {
    val m = seats.size
    val n = seats[0].size

    val allowed = IntArray(m)
    for (i in 0 until m) {
        var mask = 0
        for (j in 0 until n) {
            if (seats[i][j] == '.') {
                mask = mask or (1 shl j)
            }
        }
        allowed[i] = mask
    }

    val memo = mutableMapOf<Pair<Int, Int>, Int>()

    fun dfs(row: Int, prevMask: Int): Int {
        if (row == m) return 0
        val key = row to prevMask
        if (key in memo) return memo[key]!!

        var best = 0

        val limit = 1 shl n
        for (mask in 0 until limit) {
            if ((mask and allowed[row]) != mask) continue
            if ((mask and (mask shl 1)) != 0) continue
            if ((mask and (prevMask shl 1)) != 0) continue
            if ((mask and (prevMask shr 1)) != 0) continue

            val cur = mask.countOneBits() + dfs(row + 1, mask)
            best = maxOf(best, cur)
        }

        memo[key] = best
        return best
    }

    return dfs(0, 0)
}

fun findDuplicate(nums: IntArray): Int {
    var slow = nums[0]
    var fast = nums[nums[0]]
    while (slow != fast) {
        slow = nums[slow]
        fast = nums[nums[fast]]
    }

    slow = 0
    while (slow != fast) {
        slow = nums[slow]
        fast = nums[fast]
    }
    return slow
}

fun wonderfulSubstrings(word: String): Long {
    val n = word.length
    val seen = mutableMapOf<Int, Long>()
    seen[0] = 1L
    var status = 0
    var count = 0L
    for (i in 0 until n) {
        val mask = 1 shl (word[i] - 'a')
        status = status xor mask

        val cnt = seen[status] ?: 0L
        count += cnt


        for (c in 0 until 10) {
            val oddMask = 1 shl c
            val oddStatus = status xor oddMask
            count += (seen[oddStatus] ?: 0L)
        }
        seen[status] = cnt + 1
    }
    return count
}

fun minimumXORSum(nums1: IntArray, nums2: IntArray): Int {
    val n = nums2.size
    val limit = 1 shl n
    val dp = Array(n + 1) { IntArray(limit) { Int.MAX_VALUE } }
    dp[0][0] = 0
    for (i in 0 until n) {
        val num = nums1[i]

        for (mask in 0 until limit) {
            if (dp[i][mask] == Int.MAX_VALUE) continue

            for (j in 0 until n) {
                if ((mask shr j) and 1 == 1) continue
                val newMask = mask or (1 shl j)
                val newSum = dp[i][mask] + (num xor nums2[j])
                dp[i + 1][newMask] = minOf(dp[i + 1][newMask], newSum)
            }
        }
    }
    return dp[n][limit - 1]
}

fun singleNumber2(nums: IntArray): Int {
    var ans = 0
    for (i in 0 until 32) {
        var oneCount = 0
        for (num in nums) {
            oneCount += (num shr i) and 1
        }
        val bit = oneCount % 3
        ans = ans or (bit shl i)
    }
    return ans
}

fun singleNumber3(nums: IntArray): IntArray {
    val totalXor = nums.fold(0, Int::xor)
    val lowestDiffBit = totalXor and (-totalXor)

    var first = 0

    for (num in nums) {
        if (num and lowestDiffBit != 0) {
            first = first xor num
        }
    }
    val second = totalXor xor first
    return intArrayOf(first, second)
}

fun canSortArray(nums: IntArray): Boolean {
    var setBit = nums[0].countOneBits()
    var prevMax = -1
    var currentMin = nums[0]
    var currentMax = nums[0]
    for (i in 1 until nums.size) {
        val num = nums[i]
        val bitCount = num.countOneBits()
        if (bitCount != setBit) {
            setBit = bitCount
            if (currentMin < prevMax) return false
            prevMax = currentMax
            currentMin = num
            currentMax = num
        }
        currentMin = minOf(currentMin, num)
        currentMax = maxOf(currentMax, num)
    }
    return currentMin >= prevMax
}

fun queryString(s: String, n: Int): Boolean {
    val length = s.length
    val maxBits = 32 - n.countLeadingZeroBits()
    if (length < maxBits) return false
    println("maxBits = $maxBits")

    for (k in 1..maxBits) {
        val low = 1 shl (k - 1)
        val high = (1 shl k) - 1
        val targetCount = (minOf(n, high) - low + 1).coerceAtLeast(0)
        if (targetCount == 0) continue

        val seen = mutableSetOf<Int>()
        var num = s.takeLast(k).toInt(2)
        if (num in low..n) seen.add(num)

        var i = length - k - 1
        while (i >= 0) {
            val bit = s[i--] - '0'
            num = (num shr 1) or (bit shl (k - 1))
            if (num in low..n) seen.add(num)
            if (seen.size == targetCount) break
            //  println(num.toString(2))
        }
        if (seen.size < targetCount) return false
    }
    return true
}

fun prefixesDivBy5(nums: IntArray): List<Boolean> {
    var num = 0
    val ans = mutableListOf<Boolean>()
    for (bit in nums) {
        num = ((num shl 1) or bit) % 10
        ans.add(num % 5 == 0)
    }
    return ans
}

fun findNumOfValidWords(words: Array<String>, puzzles: Array<String>): List<Int> {
    val puzzleToMask = mutableMapOf<String, Int>()
    val puzzleGroups = mutableMapOf<Int, MutableSet<Int>>()
    val counts = mutableMapOf<Pair<Int, Int>, Int>()

    fun createMask(s: String): Int {
        var mask = 0
        for (c in s) {
            val bit = c - 'a'
            mask = mask or (1 shl bit)
        }
        return mask
    }

    for (puzzle in puzzles) {
        val id = puzzle[0] - 'a'
        val mask = createMask(puzzle)
        puzzleToMask[puzzle] = mask
        puzzleGroups.computeIfAbsent(id) { mutableSetOf() }.add(mask)
    }

    for (word in words) {
        val num = createMask(word)
        val popCount = num.countOneBits()
        if (popCount > 7) continue

        for (id in 0 until 26) {
            if ((num shr id) and 1 == 0) continue

            val masks = puzzleGroups[id] ?: continue
            for (mask in masks) {
                if (mask or num != mask) continue
                val key = id to mask
                counts[key] = (counts[key] ?: 0) + 1
            }
        }
    }


    return puzzles.map { puzzle ->
        val id = puzzle[0] - 'a'
        val mask = puzzleToMask[puzzle] ?: return@map 0
        counts[id to mask] ?: 0
    }
}

fun minFlips(a: Int, b: Int, c: Int): Int {
    var cnt = 0
    for (i in 0 until 32) {
        val bitA = (a shr i) and 1
        val bitB = (b shr i) and 1
        val bitC = (c shr i) and 1
        cnt += when {
            bitC == 0 -> bitA + bitB
            bitA == 0 && bitB == 0 -> 1
            else -> 0
        }
    }
    return cnt
}

fun sortByBits(arr: IntArray): IntArray {
    return arr.sortedBy { it.countOneBits() * 100000 + it }.toIntArray()
}

fun numSteps(s: String): Int {
    val str = s.trimStart('0')
    val n = str.length
    var cnt = 0
    var carry = 0
    var i = n - 1
    while (i > 0) {
        val bit = (str[i] - '0')  + carry

        if(bit and 1 != 0) {
            cnt ++
            carry = 1
        }
        cnt++
        i--
    }
    return cnt + carry
}

fun maxSubarrays(nums: IntArray): Int {
    val n = nums.size

    val totalAND = nums.reduce { acc, i -> acc and i }

    var pos = -1
    if (totalAND != 0) {
        var minAND = 1
        for (i in 0 until n) {
            minAND = minAND and nums[i]
            if (minAND == totalAND) {
                pos = i
                break
            }
        }
    }
    if (pos == n - 1) return 1
    println(pos)
    var cnt = 0
    var andValue = -1
    for (i in pos + 1 until n) {
        if (andValue == 0) {
            cnt++
            andValue = nums[i]
            //      println("$cnt ${i-1}")
        }
        andValue = andValue and nums[i]
    }
    if (andValue == 0) cnt++
    if (pos >= 0) cnt++
    //  println(cnt)
    return cnt
}


fun maxProduct(words: Array<String>): Int {
    val masks = words.map { word ->
        var mask = 0
        for (c in word) {
            val bit = c - 'a'
            mask = mask or (1 shl bit)
        }
        mask
    }
    val n = masks.size
    var ans = 0
    for (i in 0 until n - 1) {
        for (j in i + 1 until n) {
            if (masks[i] and masks[j] == 0) {
                ans = maxOf(ans, words[i].length * words[j].length)
            }
        }
    }
    return ans
}

fun maxProduct(s: String): Int {
    fun longestPalindromeSubseq(str: String): Int {
        val length = str.length
        if (length == 0) return 0
        val dp = Array(length) { IntArray(length) }
        for (i in length - 1 downTo 0) {
            dp[i][i] = 1
            for (j in i + 1 until length) {
                if (str[i] == str[j]) {
                    dp[i][j] = dp[i + 1][j - 1] + 2
                } else {
                    dp[i][j] = maxOf(dp[i + 1][j], dp[i][j - 1])
                }
            }
        }
        return dp[0][length - 1]
    }

    val n = s.length

    val limit = 1 shl n
    var ans = 0
    for (mask in 1 until limit - 1) {
        val first = StringBuilder()
        val second = StringBuilder()
        for (i in 0 until n) {
            val bit = (mask shr i) and 1
            if (bit == 0) first.append(s[i]) else second.append(s[i])
        }
        val firstLength = longestPalindromeSubseq(first.toString())
        val secondLength = longestPalindromeSubseq(second.toString())
        ans = maxOf(ans, firstLength * secondLength)
    }
    return ans
}

fun matrixScore(grid: Array<IntArray>): Int {
    val m = grid.size
    val n = grid[0].size

    val rows = IntArray(m)
    val cols = IntArray(n)

    for (i in 0 until m) {
        for (j in 0 until n) {
            rows[i] = rows[i] or (grid[i][j] shl (n - 1 - j))
            cols[j] = cols[j] or (grid[i][j] shl (m - 1 - i))
        }
    }

    for (i in 0 until m) {
        if (grid[i][0] == 0) {
            rows[i] = rows[i] xor ((1 shl n) - 1)
            for (j in 0 until n) {
                cols[j] = cols[j] xor (1 shl (m - 1 - i))
            }
        }
    }

    for (j in 0 until n) {
        val num = cols[j]
        val oneCount = num.countOneBits()
        if (oneCount < m - oneCount) {
            cols[j] = cols[j] xor ((1 shl m) - 1)
            for (i in 0 until m) {
                rows[i] = rows[i] xor (1 shl (n - 1 - j))
            }
        }
    }
    return rows.sum()
}

fun numberOfGoodSubsets(nums: IntArray): Int {
    val mod = 1_000_000_007L
    val primes = intArrayOf(2, 3, 5, 7, 11, 13, 17, 19, 23, 29)
    var cnt1 = nums.count { it == 1 }
    val counts = nums.filterNot { num ->
        num == 1 || primes.any { num % (it * it) == 0 }
    }.map { num ->
        var binary = 0
        for (i in 0 until primes.size) {
            if (num % primes[i] == 0) {
                binary = binary or (1 shl i)
            }
        }
        binary
    }.groupingBy { it }.eachCount()

    val maxMask = (1 shl (primes.size))
    val dp = LongArray(maxMask)
    dp[0] = 1L

    for ((num, cnt) in counts) {
        for (mask in (maxMask - 1) downTo 0) {
            if (dp[mask] <= 0) continue
            if (num and mask != 0) continue
            val newMask = num or mask
            dp[newMask] = (dp[newMask] + dp[mask] * cnt) % mod

        }
    }

    var ans = 0L
    for (mask in 1 until maxMask) {
        ans = (ans + dp[mask]) % mod
    }

    var pow2 = 1L
    while (cnt1-- > 0) pow2 = (pow2 * 2) % mod
    ans = (ans * pow2) % mod
  //  ans = (ans - 1 + mod) % mod
    return ans.toInt()
}

fun main() {
    println(
        queryString("0", 1)
    )
}