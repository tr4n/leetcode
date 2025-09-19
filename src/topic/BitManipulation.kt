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

fun main() {
    println(
        findTheLongestSubstring("eleetminicoworoep")
    )
}