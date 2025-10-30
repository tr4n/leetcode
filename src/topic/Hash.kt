package topic

import remote.UpdateRangeSmallerSegmentTree
import java.util.*
import kotlin.math.abs
import kotlin.math.sqrt

class FreqHash(val base: Long = 131L) {
    private val pow = LongArray(26) { 1L }
    var hash: Long = 0
        private set

    init {
        for (i in 1 until 26) {
            pow[i] = pow[i - 1] * base
        }
    }

    fun add(c: Char) {
        hash += pow[c - 'a']
    }

    fun remove(c: Char) {
        hash -= pow[c - 'a']
    }
}

class FreqPrefixHash(val s: String) {
    private val n = s.length
    private val base1 = 131L
    private val base2 = 137L
    private val mod1 = 1_000_000_007L
    private val mod2 = 1_000_000_009L

    private val H1 = LongArray(n + 1)
    private val H2 = LongArray(n + 1)

    init {
        val freq = IntArray(26)
        for (i in 0 until n) {
            freq[s[i] - 'a']++
            var h1 = 0L
            var h2 = 0L
            var pow1 = 1L
            var pow2 = 1L
            for (c in 0 until 26) {
                h1 = (h1 + freq[c] * pow1) % mod1
                h2 = (h2 + freq[c] * pow2) % mod2
                pow1 = (pow1 * base1) % mod1
                pow2 = (pow2 * base2) % mod2
            }
            H1[i + 1] = h1
            H2[i + 1] = h2
        }
    }

    fun hash(l: Int, r: Int): Pair<Long, Long> {
        val h1 = (H1[r + 1] - H1[l] + mod1) % mod1
        val h2 = (H2[r + 1] - H2[l] + mod2) % mod2
        return h1 to h2
    }
}


class PalindromeHasher(
    s: String,
    private var pow: LongArray = LongArray(0)
) {
    private val n = s.length
    private val base = 131L
    private val mod = 1_000_000_007L

    private val prefix = LongArray(n + 1)
    private val prefixRev = LongArray(n + 1)

    init {
        if (pow.size < n) pow = initPow(n)

        for (i in 0 until n) {
            prefix[i + 1] = (prefix[i] * base + s[i].code) % mod
        }

        for (i in n - 1 downTo 0) {
            prefixRev[n - i] = (prefixRev[n - i - 1] * base + s[i].code) % mod
        }
    }

    private fun initPow(maxLen: Int): LongArray {
        val pow = LongArray(maxLen + 1)
        pow[0] = 1L
        for (i in 0 until maxLen) {
            pow[i + 1] = (pow[i] * base) % mod
        }
        return pow
    }


    // hash  s[l..r]
    fun getHash(l: Int, r: Int): Long {
        val res = (prefix[r + 1] - (prefix[l] * pow[r - l + 1]) % mod + mod) % mod
        return res
    }

    // hash  s[l..r]
    fun getHashRev(l: Int, r: Int): Long {
        val rl = n - 1 - r
        val rr = n - 1 - l
        val res = (prefixRev[rr + 1] - (prefixRev[rl] * pow[rr - rl + 1]) % mod + mod) % mod
        return res
    }

    fun isPalindrome(l: Int, r: Int): Boolean {
        return getHash(l, r) == getHashRev(l, r)
    }
}

fun longestPalindrome1(s: String, t: String): Int {
    val hashS = PalindromeHasher(s)
    val hashT = PalindromeHasher(t)

    val sHashes = mutableSetOf<Long>()
    val tHashes = mutableSetOf<Long>()

    val palindromeStart = IntArray(s.length + 1)
    val palindromeEnd = IntArray(t.length + 1)

    var longestLength = 0
    for (i in 0 until s.length) {
        for (j in i until s.length) {
            val hash = hashS.getHash(i, j)
            val revHash = hashS.getHashRev(i, j)
            val len = j - i + 1
            sHashes.add(hash)

            val isPalindrome = hash == revHash
            if (isPalindrome) {
                longestLength = maxOf(longestLength, len)
                palindromeStart[i] = maxOf(palindromeStart[i], len)
            }
        }
    }


    for (j in (t.length - 1) downTo 0) {
        for (i in j downTo 0) {
            val hash = hashT.getHash(i, j)
            val revHash = hashT.getHashRev(i, j)
            val len = j - i + 1
            tHashes.add(hash)

            val isPalindrome = hash == revHash
            if (isPalindrome) {
                longestLength = maxOf(longestLength, len)
                palindromeEnd[j + 1] = maxOf(palindromeEnd[j + 1], len)
            }
        }
    }

    // println(palindromeStart.toList())
    //  println(palindromeEnd.toList())
    for (i in 0 until s.length) {
        for (j in i until s.length) {
            val prefix = hashS.getHash(i, j)
            // suffix in T = reverse(prefixS)
            val suffix = hashS.getHashRev(i, j)
            val prefixLen = j - i + 1

            if (suffix !in tHashes) continue
            val midPalindromeLen = palindromeStart[j + 1]
            val totalLen = 2 * prefixLen + midPalindromeLen
            longestLength = maxOf(longestLength, totalLen)
        }
    }

    for (j in (t.length - 1) downTo 0) {
        for (i in j downTo 0) {
            val suffix = hashT.getHash(i, j)
            val prefix = hashT.getHashRev(i, j)
            val suffixLen = j - i + 1
            if (prefix !in sHashes) continue

            val midPalindromeLen = palindromeEnd[i]
            //   println("${t.substring(i, j + 1)} + $midPalindromeLen")
            val totalLen = 2 * suffixLen + midPalindromeLen
            longestLength = maxOf(longestLength, totalLen)
        }
    }
    return longestLength
}

fun longestPalindrome(s: String, t: String): Int {
    val m = s.length
    val n = t.length
    val hashS = CharDoubleHasher(s)
    val hashT = CharDoubleHasher(t)

    val palindromeStart = IntArray(m)
    val palindromeEnd = IntArray(n)

    var longestLength = 0
    for (i in 0 until m) {
        for (j in i until m) {
            val hash = hashS.getHash(i, j)
            val revHash = hashS.getHashRev(i, j)
            val len = j - i + 1

            val isPalindrome = hash == revHash
            if (isPalindrome) {
                longestLength = maxOf(longestLength, len)
                palindromeStart[i] = maxOf(palindromeStart[i], len)
            }
        }
    }


    for (j in (n - 1) downTo 0) {
        for (i in j downTo 0) {
            val hash = hashT.getHash(i, j)
            val revHash = hashT.getHashRev(i, j)
            val len = j - i + 1

            val isPalindrome = hash == revHash
            if (isPalindrome) {
                longestLength = maxOf(longestLength, len)
                palindromeEnd[j] = maxOf(palindromeEnd[j], len)
            }
        }
    }

    // println(palindromeStart.toList())
    //  println(palindromeEnd.toList())
    val dp = Array(m) { IntArray(n) }
    for (i in (m - 1) downTo 0) {
        for (j in 0 until n) {
            dp[i][j] = maxOf(palindromeStart[i], palindromeEnd[j])
            if (s[i] == t[j]) {
                var innerBest = 0
                if (i + 1 < m && j - 1 >= 0) innerBest = maxOf(innerBest, dp[i + 1][j - 1])
                if (i + 1 < m) innerBest = maxOf(innerBest, palindromeStart[i + 1])
                if (j - 1 >= 0) innerBest = maxOf(innerBest, palindromeEnd[j - 1])
                dp[i][j] = maxOf(dp[i][j], 2 + innerBest)
            }
            longestLength = maxOf(longestLength, dp[i][j])
        }
    }

    //   println(dp.print())
    return longestLength
}


fun findAnagrams(s: String, p: String): List<Int> {
    if (p.length > s.length) return emptyList()
    val n = s.length
    val m = p.length

    val result = mutableListOf<Int>()
    val pFreq = FreqHash()
    for (c in p) pFreq.add(c)

    val sFreq = FreqHash()
    for (i in 0 until m) sFreq.add(s[i])

    if (sFreq.hash == pFreq.hash) result.add(0)

    for (i in 1 until n - m + 1) {
        sFreq.remove(s[i - 1])
        sFreq.add(s[i + m - 1])
        if (sFreq.hash == pFreq.hash) {
            result.add(i)
        }
    }
    return result
}

fun checkInclusion(s1: String, s2: String): Boolean {
    val n = s1.length
    val m = s2.length
    if (n > m) return false

    val firstFreq = FreqHash()
    for (c in s1) firstFreq.add(c)

    val secondFreq = FreqHash()
    for (i in 0 until n) secondFreq.add(s2[i])

    if (firstFreq.hash == secondFreq.hash) return true
    for (i in 1 until m - n + 1) {
        secondFreq.remove(s2[i - 1])
        secondFreq.add(s2[i + n - 1])
        if (secondFreq.hash == firstFreq.hash) {
            return true
        }
    }
    return false
}

fun shortestPalindrome(s: String): String {
    if (s.isEmpty()) return s
    val hasher = CharDoubleHasher(s)
    val n = s.length
    var lastIndex = 0
    for (i in n - 1 downTo 1) {
        val hash = hasher.getHash(0, i)
        val revHash = hasher.getHashRev(0, i)
        if (hash == revHash) {
            lastIndex = i
            break
        }
    }
    println(lastIndex)
    if (lastIndex == n - 1) return s
    val lead = s.substring(lastIndex + 1).reversed()

    return lead + s
}

fun longestDecomposition(text: String): Int {
    val n = text.length
    if (n == 1) return 1
    val hasher = CharDoubleHasher(text)
    val d = Array(n) { IntArray(n) }

    fun divide(start: Int, end: Int): Int {
        if (start > end) return 0
        if (start == end) return 1
        if (d[start][end] > 0) return d[start][end]
        val mid = (start + end) / 2
        var result = Int.MIN_VALUE

        for (i in 0..mid) {
            if (start + i > end - i) break
            val left = hasher.getHash(start, start + i)
            val right = hasher.getHash(end - i, end)
            if (left != right) continue

            val parts = 2 + divide(start + i + 1, end - i - 1)
            if (parts > result) {
                result = parts
            }
        }
        if (result == Int.MIN_VALUE) result = 1
        d[start][end] = result
        return result
    }
    divide(0, n - 1)
    return d[0][n - 1]
}

fun longestPrefix(s: String): String {
    val n = s.length
    if (n == 1) return ""
    val mod1 = 1_000_000_007L
    val mod2 = 1_000_000_009L
    val base1 = 131L
    val base2 = 137L
    val pow1 = LongArray(n + 1)
    val pow2 = LongArray(n + 1)
    val prefix1 = LongArray(n + 1)
    val prefix2 = LongArray(n + 1)
    pow1[0] = 1L
    pow2[0] = 1L
    for (i in 0 until n) {
        pow1[i + 1] = (pow1[i] * base1) % mod1
        pow2[i + 1] = (pow2[i] * base2) % mod2
    }
    for (i in 0 until n) {
        prefix1[i + 1] = (prefix1[i] * base1 + s[i].code) % mod1
        prefix2[i + 1] = (prefix2[i] * base2 + s[i].code) % mod2
    }

    fun getHash(l: Int, r: Int): Pair<Long, Long> {
        val hash1 = (prefix1[r + 1] - (prefix1[l] * pow1[r + 1 - l] % mod1) + mod1) % mod1
        val hash2 = (prefix2[r + 1] - (prefix2[l] * pow2[r + 1 - l] % mod2) + mod2) % mod2
        return hash1 to hash2
    }

    var index = -1
    for (i in n - 2 downTo 0) {
        val prefix = getHash(0, i)
        val suffix = getHash(n - i - 1, n - 1)
        if (suffix == prefix) {
            index = i
            break
        }
    }
    println("$index")
    return if (index < 0) "" else s.take(index + 1)
}


fun longestPalindrome(s: String): String {
    val n = s.length
    if (n == 1) return s
    val mod1 = 1_000_000_007L
    val mod2 = 1_000_000_009L
    val base1 = 131L
    val base2 = 137L
    val pow1 = LongArray(n + 1)
    val pow2 = LongArray(n + 1)
    val prefix1 = LongArray(n + 1)
    val prefix2 = LongArray(n + 1)
    val revPrefix1 = LongArray(n + 1)
    val revPrefix2 = LongArray(n + 1)
    pow1[0] = 1L
    pow2[0] = 1L
    for (i in 0 until n) {
        pow1[i + 1] = (pow1[i] * base1) % mod1
        pow2[i + 1] = (pow2[i] * base2) % mod2
    }
    for (i in 0 until n) {
        prefix1[i + 1] = (prefix1[i] * base1 + s[i].code) % mod1
        prefix2[i + 1] = (prefix2[i] * base2 + s[i].code) % mod2
    }

    for (i in n - 1 downTo 0) {
        revPrefix1[n - i] = (revPrefix1[n - i - 1] * base1 + s[i].code) % mod1
        revPrefix2[n - i] = (revPrefix2[n - i - 1] * base2 + s[i].code) % mod2

    }

    fun getHash(l: Int, r: Int): Pair<Long, Long> {
        if (l < 0 || r >= n || l > r) return 0L to 0L
        val hash1 = (prefix1[r + 1] - (prefix1[l] * pow1[r + 1 - l] % mod1) + mod1) % mod1
        val hash2 = (prefix2[r + 1] - (prefix2[l] * pow2[r + 1 - l] % mod2) + mod2) % mod2
        return hash1 to hash2
    }

    fun getRevHash(ql: Int, qr: Int): Pair<Long, Long> {
        if (ql < 0 || qr >= n || ql > qr) return 0L to 0L
        val l = n - 1 - qr
        val r = n - 1 - ql
        val hash1 = (revPrefix1[r + 1] - (revPrefix1[l] * pow1[r + 1 - l] % mod1) + mod1) % mod1
        val hash2 = (revPrefix2[r + 1] - (revPrefix2[l] * pow2[r + 1 - l] % mod2) + mod2) % mod2
        return hash1 to hash2
    }

    fun isPalindrome(l: Int, r: Int): Boolean {
        if (l < 0 || r >= n || l > r) return false
        val hash = getHash(l, r)
        val revHash = getRevHash(l, r)
        return hash == revHash
    }

    var bestL = 0
    var bestR = 0
    // odd len palindrome
    for (center in 0 until n) {
        var lo = 0
        var hi = minOf(center, n - 1 - center)

        while (lo <= hi) {
            val mid = (lo + hi) / 2
            val isPalindrome = isPalindrome(center - mid, center + mid)
            if (isPalindrome) {
                val len = 2 * mid + 1
                val bestLen = bestR - bestL + 1
                if (len > bestLen) {
                    bestL = center - mid
                    bestR = center + mid
                }
                lo = mid + 1
            } else {
                hi = mid - 1
            }
        }
    }

    // even palindrome

    for (center in 0 until n - 1) {
        var lo = 0
        var hi = minOf(center + 1, n - 2 - center)

        while (lo <= hi) {
            val mid = (lo + hi) / 2
            //  if(lo < 0)  println("$center, $lo $hi $mid")
            val isPalindrome = isPalindrome(center - mid, center + 1 + mid)
            if (isPalindrome) {
                val len = 2 * (mid + 1)
                val bestLen = bestR - bestL + 1
                if (len > bestLen) {
                    bestL = center - mid
                    bestR = center + mid + 1
                }
                lo = mid + 1
            } else {
                hi = mid - 1
            }
        }
    }
    return s.substring(bestL, bestR + 1)
}

fun maxProduct2(s: String): Long {
    val n = s.length
    if (n == 1) return 0L
    val mod1 = 1_000_000_007L
    val mod2 = 1_000_000_009L
    val base1 = 131L
    val base2 = 137L
    val pow1 = LongArray(n + 1)
    val pow2 = LongArray(n + 1)
    val prefix1 = LongArray(n + 1)
    val prefix2 = LongArray(n + 1)
    val revPrefix1 = LongArray(n + 1)
    val revPrefix2 = LongArray(n + 1)
    pow1[0] = 1L
    pow2[0] = 1L
    for (i in 0 until n) {
        pow1[i + 1] = (pow1[i] * base1) % mod1
        pow2[i + 1] = (pow2[i] * base2) % mod2
    }
    for (i in 0 until n) {
        prefix1[i + 1] = (prefix1[i] * base1 + s[i].code) % mod1
        prefix2[i + 1] = (prefix2[i] * base2 + s[i].code) % mod2
    }

    for (i in n - 1 downTo 0) {
        revPrefix1[n - i] = (revPrefix1[n - i - 1] * base1 + s[i].code) % mod1
        revPrefix2[n - i] = (revPrefix2[n - i - 1] * base2 + s[i].code) % mod2

    }

    fun getHash(l: Int, r: Int): Pair<Long, Long> {
        if (l < 0 || r >= n || l > r) return 0L to 0L
        val hash1 = (prefix1[r + 1] - (prefix1[l] * pow1[r + 1 - l] % mod1) + mod1) % mod1
        val hash2 = (prefix2[r + 1] - (prefix2[l] * pow2[r + 1 - l] % mod2) + mod2) % mod2
        return hash1 to hash2
    }

    fun getRevHash(ql: Int, qr: Int): Pair<Long, Long> {
        if (ql < 0 || qr >= n || ql > qr) return 0L to 0L
        val l = n - 1 - qr
        val r = n - 1 - ql
        val hash1 = (revPrefix1[r + 1] - (revPrefix1[l] * pow1[r + 1 - l] % mod1) + mod1) % mod1
        val hash2 = (revPrefix2[r + 1] - (revPrefix2[l] * pow2[r + 1 - l] % mod2) + mod2) % mod2
        return hash1 to hash2
    }

    fun isOddPalindrome(l: Int, r: Int): Boolean {
        if (l < 0 || r >= n || l > r) return false
        val len = r - l + 1
        if (len % 2 == 0) return false
        val hash = getHash(l, r)
        val revHash = getRevHash(l, r)
        return hash == revHash
    }

    val radius = IntArray(n)

    for (center in 0 until n) {
        var lo = 0
        var hi = minOf(center, n - 1 - center)
        var r = 0
        while (lo <= hi) {
            val mid = (lo + hi) / 2
            val isPalindrome = isOddPalindrome(center - mid, center + mid)
            if (isPalindrome) {
                r = mid
                lo = mid + 1
            } else {
                hi = mid - 1
            }
        }
        radius[center] = r
    }

    fun computePalEnds(radius: IntArray): IntArray {
        val n = radius.size
        val palEnds = IntArray(n)
        for (c in 0 until n) {
            val r = radius[c]
            val right = c + r
            if (right < n) {
                val len = 2 * r + 1
                palEnds[right] = maxOf(palEnds[right], len)
            }
        }

        for (i in n - 2 downTo 0) {
            palEnds[i] = maxOf(palEnds[i], palEnds[i + 1] - 2)
        }

        return palEnds
    }

    fun computePalStarts(radius: IntArray): IntArray {
        val n = radius.size
        val palStarts = IntArray(n)

        for (c in 0 until n) {
            val r = radius[c]
            val left = c - r
            if (left >= 0) {
                val len = 2 * r + 1
                palStarts[left] = maxOf(palStarts[left], len)
            }
        }

        for (i in 1 until n) {
            palStarts[i] = maxOf(palStarts[i], palStarts[i - 1] - 2)
        }

        return palStarts
    }

    val palStarts = computePalStarts(radius)
    val palEnds = computePalEnds(radius)

    var maxProduct = 1L
    //  println(radius.toList())
    //  println(palStarts.toList())
    //   println(palEnds.toList())
    val maxLeft = IntArray(n)
    maxLeft[0] = 1
    for (i in 1 until n) {
        maxLeft[i] = maxOf(maxLeft[i - 1], palEnds[i])
    }
    val maxRight = IntArray(n)
    maxRight[n - 1] = 1
    for (i in (n - 2) downTo 0) {
        maxRight[i] = maxOf(maxRight[i + 1], palStarts[i])
    }
    for (i in 0 until n - 1) {
        val left = maxLeft[i].toLong()
        val right = maxRight[i + 1].toLong()
        val p = left * right
        if (p > maxProduct) {
            maxProduct = p
        }
    }

    return maxProduct
}

fun longestDupSubstring(s: String): String {
    val n = s.length
    val distinctCount = s.toSet().size
    if (distinctCount == n) return ""
    if (distinctCount == 1) return s.substring(1)

    val mod1 = 1_000_000_007L
    val mod2 = 1_000_000_009L
    val base1 = 131L
    val base2 = 137L
    val pow1 = LongArray(n + 1)
    val pow2 = LongArray(n + 1)
    val prefix1 = LongArray(n + 1)
    val prefix2 = LongArray(n + 1)

    pow1[0] = 1L
    pow2[0] = 1L
    for (i in 0 until n) {
        pow1[i + 1] = (pow1[i] * base1) % mod1
        pow2[i + 1] = (pow2[i] * base2) % mod2
    }
    for (i in 0 until n) {
        prefix1[i + 1] = (prefix1[i] * base1 + s[i].code) % mod1
        prefix2[i + 1] = (prefix2[i] * base2 + s[i].code) % mod2
    }

    fun getHash(l: Int, r: Int): Pair<Long, Long> {
        if (l < 0 || r >= n || l > r) return 0L to 0L
        val hash1 = (prefix1[r + 1] - (prefix1[l] * pow1[r + 1 - l] % mod1) + mod1) % mod1
        val hash2 = (prefix2[r + 1] - (prefix2[l] * pow2[r + 1 - l] % mod2) + mod2) % mod2
        return hash1 to hash2
    }

    fun indexOfDuplicate(length: Int): Int {
        val set = mutableSetOf<Pair<Long, Long>>()
        for (i in 0 until n - length + 1) {
            val hash = getHash(i, i + length - 1)
            if (hash in set) return i
            set.add(hash)
        }
        return -1
    }

    var longest = 0
    var duplicatePos = -1
    var lo = 1
    var hi = n - 1

    while (lo <= hi) {
        val mid = (lo + hi) / 2
        val value = indexOfDuplicate(mid)
        if (value >= 0) {
            longest = mid
            duplicatePos = value
            //  println(s.substring(duplicatePos, duplicatePos + longest))
            lo = mid + 1
        } else {
            hi = mid - 1
        }
    }
    if (longest <= 0) return ""
    return s.substring(duplicatePos, duplicatePos + longest)
}

fun countMatchingSubarrays(nums: IntArray, pattern: IntArray): Int {
    val n = nums.size
    val k = pattern.size

    val numbers = IntArray(n)
    for (i in 1 until n) {
        val a = nums[i]
        val b = nums[i - 1]
        val num = when {
            a > b -> 3
            a == b -> 2
            else -> 1
        }
        numbers[i] = num
    }

    val mod = 1_000_000_007L
    val base = 5L
    val pow = LongArray(n + 1)
    val hashes = LongArray(n + 1)

    pow[0] = 1L

    for (i in 0 until n) {
        pow[i + 1] = (pow[i] * base) % mod
    }
    for (i in 0 until n) {
        hashes[i + 1] = (hashes[i] * base + numbers[i]) % mod
    }

    fun getHash(l: Int, r: Int): Long {
        if (l < 0 || r >= n || l > r) return 0L
        val hash = (hashes[r + 1] - (hashes[l] * pow[r + 1 - l] % mod) + mod) % mod
        return hash
    }

    var targetHash = 0L
    for (i in 0 until k) {
        targetHash = (targetHash * base % mod + (pattern[i] + 2L)) % mod
    }

    var cnt = 0

    for (i in 0 until n - k) {
        val hash = getHash(i + 1, i + k)
        if (hash == targetHash) cnt++
    }

    //  println(numbers.toList())
    //   println("$mod $targetHash")
    //  println(hashes.toList())
    return cnt
}

fun findRepeatedDnaSequences(s: String): List<String> {
    val n = s.length
    val mod1 = 1_000_000_007L
    val mod2 = 1_000_000_009L
    val base1 = 131L
    val base2 = 137L
    val pow1 = LongArray(n + 1)
    val pow2 = LongArray(n + 1)
    val prefix1 = LongArray(n + 1)
    val prefix2 = LongArray(n + 1)

    pow1[0] = 1L
    pow2[0] = 1L
    for (i in 0 until n) {
        pow1[i + 1] = (pow1[i] * base1) % mod1
        pow2[i + 1] = (pow2[i] * base2) % mod2
    }
    for (i in 0 until n) {
        prefix1[i + 1] = (prefix1[i] * base1 + s[i].code) % mod1
        prefix2[i + 1] = (prefix2[i] * base2 + s[i].code) % mod2
    }

    fun getHash(l: Int, r: Int): Pair<Long, Long> {
        if (l < 0 || r >= n || l > r) return 0L to 0L
        val hash1 = (prefix1[r + 1] - (prefix1[l] * pow1[r + 1 - l] % mod1) + mod1) % mod1
        val hash2 = (prefix2[r + 1] - (prefix2[l] * pow2[r + 1 - l] % mod2) + mod2) % mod2
        return hash1 to hash2
    }

    val map = mutableMapOf<Pair<Long, Long>, MutableList<Int>>()
    for (i in 0 until n - 9) {
        val hash = getHash(i, i + 9)
        if (map[hash] == null) {
            map[hash] = mutableListOf(i)
        } else {
            map[hash]?.add(i)
        }
    }

    val result = mutableListOf<String>()
    for (list in map.values) {
        if (list.size < 2) continue
        val index = list[0]
        result.add(s.substring(index, index + 10))
    }
    return result
}

fun hasAllCodes(s: String, k: Int): Boolean {
    val n = s.length
    val size = 1 shl k
    if (size > n) return false
    val used = BooleanArray(size)

    var num = 0
    for (i in 0 until k.coerceAtMost(n)) {
        num = ((num shl 1) + (s[i] - '0'))
    }

    used[num] = true
    var count = 1

    for (i in k until n) {
        num = ((num shl 1) + (s[i] - '0')) and (size - 1)
        if (!used[num]) {
            count++
            used[num] = true
            if (count == size) return true
        }

        //  println(num.toString(2))
    }

    println("$count ?? $size")
    return count == size
}

fun distinctEchoSubstrings(text: String): Int {
    val n = text.length
    val mod1 = 1_000_000_007L
    val mod2 = 1_000_000_009L
    val base1 = 131L
    val base2 = 137L
    val pow1 = LongArray(n + 1)
    val pow2 = LongArray(n + 1)
    val prefix1 = LongArray(n + 1)
    val prefix2 = LongArray(n + 1)

    pow1[0] = 1L
    pow2[0] = 1L
    for (i in 0 until n) {
        pow1[i + 1] = (pow1[i] * base1) % mod1
        pow2[i + 1] = (pow2[i] * base2) % mod2
    }
    for (i in 0 until n) {
        prefix1[i + 1] = (prefix1[i] * base1 + text[i].code) % mod1
        prefix2[i + 1] = (prefix2[i] * base2 + text[i].code) % mod2
    }

    fun getHash(l: Int, r: Int): Pair<Long, Long> {
        if (l < 0 || r >= n || l > r) return 0L to 0L
        val hash1 = (prefix1[r + 1] - (prefix1[l] * pow1[r + 1 - l] % mod1) + mod1) % mod1
        val hash2 = (prefix2[r + 1] - (prefix2[l] * pow2[r + 1 - l] % mod2) + mod2) % mod2
        return hash1 to hash2
    }

    val result = mutableSetOf<Pair<Long, Long>>()
    for (k in 1..n / 2) {
        for (i in 0 until n - 2 * k + 1) {
            val part1 = getHash(i, i + k - 1)
            val part2 = getHash(i + k, i + 2 * k - 1)
            if (part1 == part2) result.add(part1)
            // if (part1 == part2) println(text.substring(i, i + k))
        }
    }
    return result.size
}

fun longestCommonSubpath(maxNode: Int, paths: Array<IntArray>): Int {
    val m = paths.size
    val maxLen = paths.maxOf { it.size }
    paths.sortBy { it.size }

    val mod1 = 1_000_000_007L
    val mod2 = 1_000_000_009L
    val base1 = 100_003L
    val base2 = 100_007L
    val pow1 = LongArray(maxLen + 1)
    val pow2 = LongArray(maxLen + 1)

    pow1[0] = 1L
    pow2[0] = 1L
    for (i in 0 until maxLen) {
        pow1[i + 1] = (pow1[i] * base1) % mod1
        pow2[i + 1] = (pow2[i] * base2) % mod2
    }

    val prefix1 = Array(m) { LongArray(paths[it].size + 1) }
    val prefix2 = Array(m) { LongArray(paths[it].size + 1) }


    for (i in 0 until m) {
        for (j in 0 until paths[i].size) {
            prefix1[i][j + 1] = ((base1 * prefix1[i][j]) % mod1 + 1L + paths[i][j]) % mod1
            prefix2[i][j + 1] = ((base2 * prefix2[i][j]) % mod2 + 1L + paths[i][j]) % mod2
        }
    }


    fun getHash(path: Int, l: Int, r: Int): Pair<Long, Long> {
        // if (l < 0 || r >= maxLen || l > r) return 0L
        val hash1 = (prefix1[path][r + 1] - (prefix1[path][l] * pow1[r + 1 - l] % mod1) + mod1) % mod1
        val hash2 = (prefix2[path][r + 1] - (prefix2[path][l] * pow2[r + 1 - l] % mod2) + mod2) % mod2
        return Pair(hash1, hash2)
    }

    fun hasCommon(len: Int): Boolean {
        var set = mutableSetOf<Pair<Long, Long>>()

        for (i in 0 until paths[0].size - len + 1) {
            //  println(paths[0].toList().subList(i, i + len))
            val hash = getHash(0, i, i + len - 1)
            set.add(hash)
        }

        for (path in 1 until m) {
            val newSet = mutableSetOf<Pair<Long, Long>>()

            for (i in 0 until paths[path].size - len + 1) {
                val hash = getHash(path, i, i + len - 1)
                if (hash in set) newSet.add(hash)
            }
            if (newSet.isEmpty()) return false
            set = newSet
        }
//        if (set.isNotEmpty()) {
//            println("len $len, $set")
//        }
        return set.isNotEmpty()
    }

    val firstPathSize = paths[0].size
    var longest = 0
    var lo = 1
    var hi = firstPathSize

    while (lo <= hi) {
        val len = (lo + hi) / 2

        val hasCommon = hasCommon(len)

        if (hasCommon) {
            longest = len
            lo = len + 1
        } else {
            hi = len - 1
        }
    }
    return longest
}

fun sumScores(s: String): Long {
    val n = s.length
    val hasher = CharDoubleHasher(s)

    var sum = 0L
    for (i in (n - 1) downTo 0) {
        var score = 0
        var lo = 1
        var hi = n - i

        while (lo <= hi) {
            val mid = (lo + hi) / 2
            val prefix1 = hasher.getHash(0, mid - 1)
            val prefix2 = hasher.getHash(i, i + mid - 1)
            if (prefix1 == prefix2) {
                score = mid
                lo = mid + 1
            } else {
                hi = mid - 1
            }
        }
        sum += score.toLong()
    }
    return sum
}

fun beautifulIndices(s: String, a: String, b: String, k: Int): List<Int> {
    val n = s.length
    val mod1 = 1_000_000_007L
    val mod2 = 1_000_000_009L
    val base1 = 131L
    val base2 = 137L
    val pow1 = LongArray(n + 1)
    val pow2 = LongArray(n + 1)
    val prefix1 = LongArray(n + 1)
    val prefix2 = LongArray(n + 1)

    pow1[0] = 1L
    pow2[0] = 1L
    for (i in 0 until n) {
        pow1[i + 1] = (pow1[i] * base1) % mod1
        pow2[i + 1] = (pow2[i] * base2) % mod2
    }
    for (i in 0 until n) {
        prefix1[i + 1] = (prefix1[i] * base1 + s[i].code) % mod1
        prefix2[i + 1] = (prefix2[i] * base2 + s[i].code) % mod2
    }

    fun getHash(l: Int, r: Int): Pair<Long, Long> {
        if (l < 0 || r >= n || l > r) return 0L to 0L
        val hash1 = (prefix1[r + 1] - (prefix1[l] * pow1[r + 1 - l] % mod1) + mod1) % mod1
        val hash2 = (prefix2[r + 1] - (prefix2[l] * pow2[r + 1 - l] % mod2) + mod2) % mod2
        return hash1 to hash2
    }

    var hashA1 = 0L
    var hashA2 = 0L
    for (i in 0 until a.length) {
        hashA1 = (hashA1 * base1 + a[i].code) % mod1
        hashA2 = (hashA2 * base2 + a[i].code) % mod2
    }

    var hashB1 = 0L
    var hashB2 = 0L
    for (i in 0 until b.length) {
        hashB1 = (hashB1 * base1 + b[i].code) % mod1
        hashB2 = (hashB2 * base2 + b[i].code) % mod2
    }
    val hashA = Pair(hashA1, hashA2)
    val hashB = Pair(hashB1, hashB2)

    val posA = mutableListOf<Int>()
    val posB = mutableListOf<Int>()
    for (i in 0 until n) {
        val rightA = i + a.length - 1
        val rightB = i + b.length - 1
        if (rightA < n && getHash(i, rightA) == hashA) posA.add(i)
        if (rightB < n && getHash(i, rightB) == hashB) posB.add(i)
    }

    val result = mutableListOf<Int>()
    for (i in posA) {
        val a = (i - k).coerceAtLeast(0)
        val b = (i + k).coerceAtMost(n - 1)
        val idx = posB.binarySearch(a).let { if (it < 0) -it - 1 else it }
        val hasJ = idx < posB.size && posB[idx] <= b
        if (hasJ) result.add(i)
    }
    return result
}

class DoubleRollingHash(
    s: String,
    private val base1: Long = 131L,
    private val mod1: Long = 1_000_000_007L,
    private val base2: Long = 137L,
    private val mod2: Long = 1_000_000_009L
) {
    private val n = s.length
    private val prefix1 = LongArray(n + 1)
    private val prefix2 = LongArray(n + 1)
    private val pow1 = LongArray(n + 1)
    private val pow2 = LongArray(n + 1)

    init {
        pow1[0] = 1
        pow2[0] = 1
        for (i in 1..n) {
            pow1[i] = (pow1[i - 1] * base1) % mod1
            pow2[i] = (pow2[i - 1] * base2) % mod2
            prefix1[i] = (prefix1[i - 1] * base1 + s[i - 1].code) % mod1
            prefix2[i] = (prefix2[i - 1] * base2 + s[i - 1].code) % mod2
        }
    }

    fun getHash(l: Int, r: Int): Pair<Long, Long> {
        val h1 = (prefix1[r + 1] - (prefix1[l] * pow1[r - l + 1]) % mod1 + mod1) % mod1
        val h2 = (prefix2[r + 1] - (prefix2[l] * pow2[r - l + 1]) % mod2 + mod2) % mod2
        return Pair(h1, h2)
    }
}

class DoubleHash(
    private val base1: Long = 131L,
    private val mod1: Long = 1_000_000_007L,
    private val base2: Long = 137L,
    private val mod2: Long = 1_000_000_009L
) {
    fun hash(s: String): Pair<Long, Long> {
        var h1 = 0L
        var h2 = 0L
        for (c in s) {
            h1 = (h1 * base1 + c.code) % mod1
            h2 = (h2 * base2 + c.code) % mod2
        }
        return Pair(h1, h2)
    }
}

class CharDoubleHasher(
    s: String,
    private var pow1: LongArray = LongArray(0),
    private var pow2: LongArray = LongArray(0)
) {
    private val n = s.length
    private val base1 = 131L
    private val mod1 = 1_000_000_007L
    private val base2 = 137L
    private val mod2 = 1_000_000_009L

    private val prefix1 = LongArray(n + 1)
    private val prefix2 = LongArray(n + 1)
    private val prefixRev1 = LongArray(n + 1)
    private val prefixRev2 = LongArray(n + 1)

    init {
        if (pow1.size < n) pow1 = initPow(n, base1, mod1)
        if (pow2.size < n) pow2 = initPow(n, base2, mod2)

        for (i in 0 until n) {
            prefix1[i + 1] = (prefix1[i] * base1 + s[i].code) % mod1
            prefix2[i + 1] = (prefix2[i] * base2 + s[i].code) % mod2
        }

        for (i in n - 1 downTo 0) {
            prefixRev1[n - i] = (prefixRev1[n - i - 1] * base1 + s[i].code) % mod1
            prefixRev2[n - i] = (prefixRev2[n - i - 1] * base2 + s[i].code) % mod2
        }
    }

    private fun initPow(maxLen: Int, base: Long, mod: Long): LongArray {
        val pow = LongArray(maxLen + 1)
        pow[0] = 1L
        for (i in 0 until maxLen) {
            pow[i + 1] = (pow[i] * base) % mod
        }
        return pow
    }

    // hash s[l..r]
    fun getHash(l: Int, r: Int): Pair<Long, Long> {
        val h1 = (prefix1[r + 1] - (prefix1[l] * pow1[r - l + 1]) % mod1 + mod1) % mod1
        val h2 = (prefix2[r + 1] - (prefix2[l] * pow2[r - l + 1]) % mod2 + mod2) % mod2
        return Pair(h1, h2)
    }

    // hash reverse s[l..r]
    fun getHashRev(ll: Int, rr: Int): Pair<Long, Long> {
        val l = n - 1 - rr
        val r = n - 1 - ll
        val h1 = (prefixRev1[r + 1] - (prefixRev1[l] * pow1[r - l + 1]) % mod1 + mod1) % mod1
        val h2 = (prefixRev2[r + 1] - (prefixRev2[l] * pow2[r - l + 1]) % mod2 + mod2) % mod2
        return Pair(h1, h2)
    }
}


fun minValidStrings(words: Array<String>, target: String): Int {

    val mod1 = 1_000_000_007L
    val mod2 = 1_000_000_009L
    val base1 = 131L
    val base2 = 137L

    val prefixes = mutableSetOf<Pair<Long, Long>>()
    for (word in words) {
        var hash1 = 0L
        var hash2 = 0L

        for (c in word) {
            hash1 = (hash1 * base1 + c.code) % mod1
            hash2 = (hash2 * base2 + c.code) % mod2
            prefixes.add(Pair(hash1, hash2))
        }
    }

    val maxLen = words.maxOf { it.length }
    val hasher = CharDoubleHasher(target)

    val n = target.length
    val dp = IntArray(n + 1) { Int.MAX_VALUE }
    dp[0] = 0
    val tree = UpdateRangeSmallerSegmentTree(dp)

    for (i in 0 until n) {
        dp[i] = tree.query(i, i)
        if (dp[i] == Int.MAX_VALUE) continue
        var l = i
        var r = minOf(i + maxLen - 1, n - 1)
        var j = -1
        while (l <= r) {
            val mid = (l + r) / 2
            val hash = hasher.getHash(i, mid)
            if (hash in prefixes) {
                j = mid
                l = mid + 1
            } else {
                r = mid - 1
            }
        }
        if (j < 0) continue
        //  println(target.subSequence(i, j + 1))
        tree.updateRange(i + 1, j + 1, dp[i] + 1)
    }
    dp[n] = tree.query(n, n)
    //  println(dp.toList())
    return if (dp[n] == Int.MAX_VALUE) -1 else dp[n]
}

fun deleteString(s: String): Int {
    val n = s.length
    val hasher = CharDoubleHasher(s)

    val dp = IntArray(n) { Int.MIN_VALUE }
    dp[0] = 1
    for (i in 0 until n) {

        for (len in 1..(n - i) / 2) {
            val preHash = hasher.getHash(i, i + len - 1)
            val forHash = hasher.getHash(i + len, i + 2 * len - 1)
            if (preHash != forHash) continue
            dp[i + len] = maxOf(dp[i] + 1, dp[i + len])
        }
    }
    println(dp.toList())
    return dp.max()
}

fun minimumTimeToInitialState(word: String, k: Int): Int {
    val n = word.length
    val hasher = CharDoubleHasher(word)

    var cut = k
    while (cut < n) {
        //  println(word.substring(i))
        val hashToEnd = hasher.getHash(cut, n - 1)
        val prefixHash = hasher.getHash(0, n - 1 - cut)
        if (hashToEnd == prefixHash) return cut / k
        cut += k
    }
    return cut / k
}

fun minimumCost(target: String, words: Array<String>, costs: IntArray): Int {

    val hashFactory = DoubleHash()
    val hasher = DoubleRollingHash(target)

    val wordsByLen = TreeMap<Int, MutableSet<Pair<Long, Long>>>()
    val costsByHash = mutableMapOf<Pair<Long, Long>, Int>()
    for (i in words.indices) {
        val word = words[i]
        val hash = hashFactory.hash(word)
        wordsByLen.computeIfAbsent(word.length) { mutableSetOf() }.add(hash)
        costsByHash[hash] = minOf(costsByHash[hash] ?: Int.MAX_VALUE, costs[i])
    }


    val n = target.length
    val dp = IntArray(n + 1) { Int.MAX_VALUE }
    dp[0] = 0

    for (i in 0 until n) {
        if (dp[i] == Int.MAX_VALUE) continue
        val maxLen = n - i
        for ((len, set) in wordsByLen.headMap(maxLen, true)) {
            val j = i + len - 1
            val hash = hasher.getHash(i, j)
            if (hash !in set) continue
            val cost = costsByHash[hash] ?: continue
            dp[j + 1] = minOf(dp[j + 1], dp[i] + cost)
        }
    }

    return if (dp[n] == Int.MAX_VALUE) -1 else dp[n]
}

fun countCells(grid: Array<CharArray>, pattern: String): Int {
    val m = grid.size
    val n = grid[0].size
    val total = m * n
    val len = pattern.length
    if (len > total) return 0

    val base1 = 131L
    val mod1 = 1_000_000_007L
    val base2 = 137L
    val mod2 = 1_000_000_009L

    val pow1 = LongArray(len) { 1 }
    val pow2 = LongArray(len) { 1 }
    for (i in 1 until len) {
        pow1[i] = (pow1[i - 1] * base1) % mod1
        pow2[i] = (pow2[i - 1] * base2) % mod2
    }

    // Pattern hash
    var pat1 = 0L
    var pat2 = 0L
    for (c in pattern) {
        pat1 = (pat1 * base1 + c.code) % mod1
        pat2 = (pat2 * base2 + c.code) % mod2
    }


    fun scan(isRow: Boolean, diff: IntArray) {
        var h1 = 0L
        var h2 = 0L
        for (i in 0 until len) {
            val (r, c) = if (isRow) i / n to i % n else i % m to i / m
            h1 = (h1 * base1 + grid[r][c].code) % mod1
            h2 = (h2 * base2 + grid[r][c].code) % mod2
        }
        if (h1 == pat1 && h2 == pat2) {
            diff[0]++
            diff[len]--
        }

        for (i in len until total) {
            val (rOut, cOut) = if (isRow) (i - len) / n to (i - len) % n else (i - len) % m to (i - len) / m
            h1 = (h1 - grid[rOut][cOut].code * pow1[len - 1] % mod1 + mod1) % mod1
            h2 = (h2 - grid[rOut][cOut].code * pow2[len - 1] % mod2 + mod2) % mod2

            val (rIn, cIn) = if (isRow) i / n to i % n else i % m to i / m
            h1 = (h1 * base1 + grid[rIn][cIn].code) % mod1
            h2 = (h2 * base2 + grid[rIn][cIn].code) % mod2

            if (h1 == pat1 && h2 == pat2) {
                diff[i - len + 1]++
                diff[i + 1]--
            }
        }
    }

    val rowDiff = IntArray(total + 1)
    val colDiff = IntArray(total + 1)

    scan(true, rowDiff)
    scan(false, colDiff)

    var cnt = 0
    val rowMark = BooleanArray(total)
    for (i in 0 until total) {
        cnt += rowDiff[i]
        if (cnt > 0) rowMark[i] = true
    }

    cnt = 0
    val colMark = BooleanArray(total)
    for (i in 0 until total) {
        cnt += colDiff[i]
        if (cnt > 0) {
            val col = i / m
            val row = i % m
            val id = row * n + col
            colMark[id] = true
        }
    }

    var result = 0
    for (id in 0 until total) if (rowMark[id] && colMark[id]) result++
    return result
}

fun countSubstrings(s: String): Int {
    class CharDoubleHasher(
        s: String,
        private var pow1: LongArray = LongArray(0),
        private var pow2: LongArray = LongArray(0)
    ) {
        private val n = s.length
        private val base1 = 131L
        private val mod1 = 1_000_000_007L
        private val base2 = 137L
        private val mod2 = 1_000_000_009L

        private val prefix1 = LongArray(n + 1)
        private val prefix2 = LongArray(n + 1)
        private val prefixRev1 = LongArray(n + 1)
        private val prefixRev2 = LongArray(n + 1)

        init {
            if (pow1.size < n) pow1 = initPow(n, base1, mod1)
            if (pow2.size < n) pow2 = initPow(n, base2, mod2)

            for (i in 0 until n) {
                prefix1[i + 1] = (prefix1[i] * base1 + s[i].code) % mod1
                prefix2[i + 1] = (prefix2[i] * base2 + s[i].code) % mod2
            }

            for (i in n - 1 downTo 0) {
                prefixRev1[n - i] = (prefixRev1[n - i - 1] * base1 + s[i].code) % mod1
                prefixRev2[n - i] = (prefixRev2[n - i - 1] * base2 + s[i].code) % mod2
            }
        }

        private fun initPow(maxLen: Int, base: Long, mod: Long): LongArray {
            val pow = LongArray(maxLen + 1)
            pow[0] = 1L
            for (i in 0 until maxLen) {
                pow[i + 1] = (pow[i] * base) % mod
            }
            return pow
        }

        // hash s[l..r]
        fun getHash(l: Int, r: Int): Pair<Long, Long> {
            val h1 = (prefix1[r + 1] - (prefix1[l] * pow1[r - l + 1]) % mod1 + mod1) % mod1
            val h2 = (prefix2[r + 1] - (prefix2[l] * pow2[r - l + 1]) % mod2 + mod2) % mod2
            return Pair(h1, h2)
        }

        // hash reverse s[l..r]
        fun getHashRev(l: Int, r: Int): Pair<Long, Long> {
            val rl = n - 1 - r
            val rr = n - 1 - l
            val h1 = (prefixRev1[rr + 1] - (prefixRev1[rl] * pow1[rr - rl + 1]) % mod1 + mod1) % mod1
            val h2 = (prefixRev2[rr + 1] - (prefixRev2[rl] * pow2[rr - rl + 1]) % mod2 + mod2) % mod2
            return Pair(h1, h2)
        }
    }

    val hasher = CharDoubleHasher(s)
    val n = s.length
    var cnt = n
    for (i in 0 until n) {
        for (j in i + 1 until n) {
            val h1 = hasher.getHash(i, j)
            val h2 = hasher.getHashRev(i, j)
            if (h1 == h2) cnt++
        }
    }
    return cnt
}

fun minSteps(n: Int): Int {
//    val dp = Array(n + 1) { IntArray(n + 1) { n + 3 } }
//    if (n == 1) return 0
//    dp[1][1] = 1
//    for (i in 2..n) {
//        if (i % 2 == 0) {
//            dp[i][i / 2] = minOf(dp[i][i / 2], dp[i / 2].min() + 2)
//        }
//        for (j in 1 until i) {
//            dp[i][j] = minOf(dp[i][j], dp[i - j][j] + 1)
//
//        }
//    }
//    return dp[n].min()
    var num = n
    var ans = 0
    var d = 2
    while (d * d <= num) {
        while (num % d == 0) {
            ans += d
            num /= d
        }
        d++
    }
    if (num > 1) ans += num
    return ans
}

fun smallestValue(n: Int): Int {
    var divisorSum = 0
    var number = n
    while (true) {
        var num = number
        divisorSum = 0
        var d = 2
        while (d * d <= num) {
            while (num % d == 0) {
                divisorSum += d
                num /= d
            }
            d++
        }
        if (num > 1) divisorSum += num
        if (divisorSum == number) return number
        number = divisorSum
    }
    return n
}

fun distinctPrimeFactors(nums: IntArray): Int {
    val primes = mutableSetOf<Int>()

    for (num in nums) {
        if (num <= 3) {
            primes.add(num)
            continue
        }
        var number = num
        while (number % 2 == 0) {
            primes.add(2)
            number /= 2
        }
        val sqrtNum = sqrt(number.toDouble()).toInt()
        for (d in 3..sqrtNum step 2) {
            while (number % d == 0) {
                primes.add(d)
                number /= d
            }
        }
        if (number > 1) primes.add(number)
    }
    return primes.size
}

fun closestDivisors(num: Int): IntArray {
    var a1 = sqrt((num + 1).toDouble()).toInt()
    while (a1 > 0 && (num + 1) % a1 != 0) a1--
    val delta1 = if (a1 > 0) abs(a1 - (num + 1) / a1) else Int.MAX_VALUE

    var b1 = sqrt((num + 2).toDouble()).toInt()
    while (b1 > 0 && (num + 2) % b1 != 0) b1--
    val delta2 = if (b1 > 0) abs(b1 - (num + 2) / b1) else Int.MAX_VALUE
    return if (delta1 < delta2) {
        intArrayOf(a1, (num + 1) / a1)
    } else {
        intArrayOf(b1, (num + 2) / b1)
    }
}

fun findReplaceString(s: String, indices: IntArray, sources: Array<String>, targets: Array<String>): String {
    val n = s.length
    val mod1 = 1_000_000_007L
    val mod2 = 1_000_000_009L
    val base1 = 131L
    val base2 = 137L
    val pow1 = LongArray(n + 1)
    val pow2 = LongArray(n + 1)
    pow1[0] = 1L
    pow2[0] = 1L
    for (i in 1..n) {
        pow1[i] = (pow1[i - 1] * base1) % mod1
        pow2[i] = (pow2[i - 1] * base2) % mod2
    }

    val prefix1 = LongArray(n + 1)
    val prefix2 = LongArray(n + 1)
    for (i in 0 until n) {
        val code = s[i].code
        prefix1[i + 1] = (prefix1[i] * base1 + code) % mod1
        prefix2[i + 1] = (prefix2[i] * base2 + code) % mod2
    }

    fun getHash(l: Int, r: Int): Pair<Long, Long> {
        if (l < 0 || r < 0 || l > r) return 0L to 0L
        val hash1 = (prefix1[r + 1] - (prefix1[l] * pow1[r - l + 1] % mod1) + mod1) % mod1
        val hash2 = (prefix2[r + 1] - (prefix2[l] * pow2[r - l + 1] % mod2) + mod2) % mod2
        return hash1 to hash2
    }

    val builder = StringBuilder()
    var j = 0

    val triples = (0 until indices.size).map {
        Triple(indices[it], sources[it], targets[it])
    }.sortedBy { it.first }

    for (i in 0 until triples.size) {
        val (index, source, target) = triples[i]

        val length = source.length
        if (index + length > n) continue

        val subHash = getHash(index, index + length - 1)

        while (j < index) builder.append(s[j++])

        var sourceHash1 = 0L
        var sourceHash2 = 0L
        for (c in source) {
            val code = c.code
            sourceHash1 = (sourceHash1 * base1 + code) % mod1
            sourceHash2 = (sourceHash2 * base2 + code) % mod2
        }
        val sourceHash = sourceHash1 to sourceHash2

        if (sourceHash != subHash) {
            continue
        }
        builder.append(target)
        j += length
    }

    while (j < s.length) builder.append(s[j++])
    return builder.toString()
}

fun longestValidSubstring(word: String, forbidden: List<String>): Int {
    val blocks = mutableSetOf<String>()
    val lens = mutableSetOf<Int>()

    for (str in forbidden) {
        blocks.add(str)
        lens.add(str.length)
    }
    val suffixLens = lens.sorted()
    //  println(suffixLens)
    val n = word.length
    var l = 0
    var r = 0
    var maxLen = 0
    while (r < n) {
        val length = r - l + 1
        for (len in suffixLens) {
            if (len > length) break
            val suffix = word.substring(r - len + 1, r + 1)

            if (suffix !in blocks) continue

            maxLen = maxOf(maxLen, r - l)
            l = r - len + 2
            //  println("$suffix $l $r")
            break
        }
        maxLen = maxOf(maxLen, r - l + 1)
        r++

    }
    return maxLen
}


fun validSubstringCount(word1: String, word2: String): Long {
    val m = word2.length
    val n = word1.length
    if (m > n) return 0L

    val target = IntArray(26)
    var targetMask = 0
    for (c in word2) {
        val id = c - 'a'
        target[id]++
        targetMask = targetMask or (1 shl id)
    }


    var mask = 0
    var l = 0
    val freq = IntArray(26)
    var ans = 0L
    for (r in 0 until n) {
        val c = word1[r] - 'a'
        freq[c]++
        if (freq[c] >= target[c]) {
            mask = mask or (1 shl c)
        }

        while (mask and targetMask == targetMask) {
            ans += (n - r).toLong()
            val first = word1[l] - 'a'
            freq[first]--
            if (freq[first] < target[first]) {
                mask = mask and (1 shl first).inv()
            }
            l++
        }
    }

    return ans
}

fun subStrHash(s: String, power: Int, module: Int, k: Int, hashValue: Int): String {
    val n = s.length
    val target = hashValue.toLong()

    // reverse str
    val value = LongArray(n) { s[n - it - 1] - 'a' + 1L }

    // pre-compute pow^(k -1)
    var pow = 1L
    for (i in 1 until k) pow = pow * power % module

    var pos = 0
    var hash = 0L
    for (i in 0 until k){
        hash = (hash * power + value[i]) % module
    }

    if (hash == target) {
        pos = k - 1
    }

    for (i in k until n) {
        val first = value[i - k]
        val last = value[i]
        hash = (hash - first * pow % module + module) % module
        hash = (hash * power + last) % module
        if (hash == target) pos = i

    }
    val start = n - 1 - pos
    return s.substring(start, start + k)
}

fun main() {
    println(
        minSteps(3)
    )
}
