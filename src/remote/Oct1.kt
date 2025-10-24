package remote

fun countCoveredBuildings(n: Int, buildings: Array<IntArray>): Int {
    val rows = mutableMapOf<Int, Pair<Int, Int>>()
    val cols = mutableMapOf<Int, Pair<Int, Int>>()
    var cnt = 0
    val minValue = -100_005
    val maxValue = 100_005

    for ((x, y) in buildings) {
        val (minY, maxY) = rows[x] ?: (maxValue to minValue)
        val (minX, maxX) = cols[y] ?: (maxValue to minValue)
        if (x in (minX + 1)..<maxX && y in (minY + 1)..<maxY) cnt++
        rows[x] = minOf(minY, y) to maxOf(maxY, y)
        cols[y] = minOf(minX, x) to maxOf(maxX, x)
    }
    return cnt
}

fun maxPartitionsAfterOperations(s: String, k: Int): Int {
    val n = s.length
    val left = Array(n) { IntArray(3) }
    val right = Array(n) { IntArray(3) }

    var num = 0
    var mask = 0
    var count = 0
    for (i in 0 until n - 1) {
        val binary = 1 shl (s[i] - 'a')
        if ((mask and binary) == 0) {
            count++
            if (count <= k) {
                mask = mask or binary
            } else {
                num++
                mask = binary
                count = 1
            }
        }
        left[i + 1][0] = num
        left[i + 1][1] = mask
        left[i + 1][2] = count
    }

    num = 0
    mask = 0
    count = 0
    for (i in n - 1 downTo 1) {
        val binary = 1 shl (s[i] - 'a')
        if ((mask and binary) == 0) {
            count++
            if (count <= k) {
                mask = mask or binary
            } else {
                num++
                mask = binary
                count = 1
            }
        }
        right[i - 1][0] = num
        right[i - 1][1] = mask
        right[i - 1][2] = count
    }

    var maxVal = 0
    for (i in 0 until n) {
        var seg = left[i][0] + right[i][0] + 2

        val totMask = left[i][1] or right[i][1]
        val totCount = totMask.countOneBits()

        if (left[i][2] == k && right[i][2] == k && totCount < 26) {
            seg++
        } else if (minOf(totCount + 1, 26) <= k) {
            seg--
        }
        maxVal = maxOf(maxVal, seg)
    }
    return maxVal
}

fun maximumsSplicedArray(nums1: IntArray, nums2: IntArray): Int {
    val numbers1 = nums1.map { it.toLong() }
    val numbers2 = nums2.map { it.toLong() }
    val n = numbers1.size
    val prefix = LongArray(n + 1)
    for (i in 0 until n) prefix[i + 1] = prefix[i] + numbers1[i] - numbers2[i]

    val suffixMin = LongArray(n + 1)
    suffixMin[n] = Int.MAX_VALUE.toLong()
    for (i in (n - 1) downTo 0) suffixMin[i] = minOf(suffixMin[i + 1], prefix[i + 1])

    val s1 = numbers1.sum()
    val s2 = numbers2.sum()

    var ans = maxOf(s1, s2)
    var leftMin = Int.MAX_VALUE.toLong()
    for (i in 0 until n) {
        val center = prefix[i + 1]
        leftMin = minOf(leftMin, prefix[i])
        ans = maxOf(s2 + center - leftMin, ans)

        val rightMin = suffixMin[i + 1]
        ans = maxOf(s1 + center - rightMin, ans)
    }
    return ans.toInt()
}

fun checkArithmeticSubarrays(arr: IntArray, left: IntArray, right: IntArray): List<Boolean> {

    fun isArithmeticSubarray(l: Int, r: Int): Boolean {
        val len = r - l + 1
        if (len <= 2) return true

        var minVal = Int.MAX_VALUE
        var maxVal = Int.MIN_VALUE
        for (i in l..r) {
            minVal = minOf(minVal, arr[i])
            maxVal = maxOf(maxVal, arr[i])
        }

        if ((maxVal - minVal) % (len - 1) != 0) return false
        val d = (maxVal - minVal) / (len - 1)
        if (d == 0) {
            for (i in l..r) if (arr[i] != minVal) return false
            return true
        }

        val seen = mutableSetOf<Int>()
        for (i in l..r) {
            val diff = arr[i] - minVal
            if (diff % d != 0) return false
            val pos = diff / d
            if (!seen.add(pos)) return false
        }
        return true
    }

    return MutableList(left.size) {
        isArithmeticSubarray(left[it], right[it])
    }
}

fun pivotArray(nums: IntArray, pivot: Int): IntArray {
    val n = nums.size
    val smaller = mutableListOf<Int>()
    val same = mutableListOf<Int>()
    val greater = mutableListOf<Int>()

    for (num in nums) {
        if (num > pivot) {
            greater.add(num)
        } else if (num < pivot) {
            smaller.add(num)
        } else same.add(num)
    }
    return (smaller + same + greater).toIntArray()
}

fun groupThePeople(groupSizes: IntArray): List<List<Int>> {
    val groups = mutableMapOf<Int, MutableList<Int>>()
    for (i in groupSizes.indices) {
        val size = groupSizes[i]
        groups.computeIfAbsent(size) { mutableListOf() }.add(i)
    }
    val result = mutableListOf<List<Int>>()
    for ((size, list) in groups) {
        for (i in 0 until list.size step size) {
            result.add(list.subList(i, i + size))
        }
    }
    return result
}

fun smallestDivisor(nums: IntArray, threshold: Int): Int {
    val limit = threshold.toLong()
    var lo = 1
    var hi = nums.max()
    var ans = hi
    while (lo <= hi) {
        val mid = (lo + hi) / 2
        val sum = nums.sumOf { (it.toLong() + mid - 1) / mid }
        if (sum <= limit) {
            ans = mid
            hi = mid - 1
        } else lo = mid + 1
    }
    return ans
}

class DistinctST(private val data: IntArray) {
    private val n = data.size
    private val tree = Array(4 * n) {
        Array(2) { mutableMapOf<Int, Int>() }
    }

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            val num = data[l]
            tree[node][num % 2][num] = 1
            return
        }
        val mid = (l + r) / 2
        build(2 * node, l, mid)
        build(2 * node + 1, mid + 1, r)
        tree[node][0] = merge(tree[2 * node][0], tree[2 * node + 1][0])
        tree[node][1] = merge(tree[2 * node][1], tree[2 * node + 1][1])
    }

    private fun merge(map1: MutableMap<Int, Int>, map2: MutableMap<Int, Int>): MutableMap<Int, Int> {
        val map = mutableMapOf<Int, Int>()
        for ((key, value) in map2) {
            map[key] = (map[key] ?: 0) + value
        }
        for ((key, value) in map1) {
            map[key] = (map[key] ?: 0) + value
        }
        return map
    }

    private fun query(node: Int, l: Int, r: Int, ql: Int, qr: Int): Int {
        if (l > qr || r < ql) return 0
        if (l >= ql && r <= qr) return tree[node].size
        val mid = (l + r) / 2
        return query(2 * node, l, mid, ql, qr) + query(2 * node + 1, mid + 1, r, ql, qr)
    }
}

fun longestBalanced(nums: IntArray): Int {
    val n = nums.size

    var ans = 0
    for (i in 0 until n) {
        if (n - i <= ans) break
        val map = mutableMapOf<Int, Int>()
        val count = IntArray(2)

        for (j in i until n) {
            val num = nums[j]
            val cnt = (map[num] ?: 0) + 1
            map[num] = cnt
            if (cnt == 1) {
                count[num % 2]++
            }
            if (count[0] == count[1]) {
                ans = maxOf(ans, j - i + 1)
            }
        }
    }

    return ans
}

fun nextBeautifulNumber(n: Int): Int {
    var numDigits = 0

    var x = n
    while(x > 0) {
        numDigits ++
        x/=10
    }

    var count = IntArray(10)
    var ans = Long.MAX_VALUE
    fun backtrack(pos: Int, num: Long) {
        if(ans != Long.MAX_VALUE) return

        if(pos == numDigits) {
            if(num <= n) return
            for(d in 1..9) {
                val cnt = count[d]
                if(cnt > 0 && cnt != d) return
            }
            ans = minOf(ans, num)
            return
        }

        for(d in 1..9) {
            if(count[d] >= d) continue
            val newNum = num * 10L + d
            count[d]++
            backtrack(pos + 1, newNum)
            count[d]--
        }
    }
    while(ans == Long.MAX_VALUE) {
        count = IntArray(10)
        backtrack(0, 0L)
        numDigits ++
    }
    return ans.toInt()
}