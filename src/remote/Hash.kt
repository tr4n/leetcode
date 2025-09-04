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

