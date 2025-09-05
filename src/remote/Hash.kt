package remote

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

class PalindromeDoubleHasher(
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

fun longestPalindrome(s: String, t: String): Int {
    val m = s.length
    val n = t.length
    val hashS = PalindromeDoubleHasher(s)
    val hashT = PalindromeDoubleHasher(t)

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

fun main() {
    println(
        longestPalindrome("mrb", "r")
    )
}
