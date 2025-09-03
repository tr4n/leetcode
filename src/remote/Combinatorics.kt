package remote

class Combinatorics(nMax: Int, val mod: Long = 1_000_000_007L) {
    private val fact = LongArray(nMax + 1)
    private val invFact = LongArray(nMax + 1)

    init {
        // fact[0] = 1
        fact[0] = 1
        for (i in 1..nMax) {
            fact[i] = (fact[i - 1] * i) % mod
        }

        // invFact[nMax] = (fact[nMax])^(mod-2)
        invFact[nMax] = modPow(fact[nMax], mod - 2, mod)
        for (i in nMax downTo 1) {
            invFact[i - 1] = (invFact[i] * i) % mod
        }
    }

    fun modPow(base: Long, exp: Long, mod: Long = this.mod): Long {
        var b = base % mod
        var e = exp
        var res = 1L
        while (e > 0) {
            if (e and 1L == 1L) res = (res * b) % mod
            b = (b * b) % mod
            e = e shr 1
        }
        return res
    }

    // n!
    fun fact(n: Int): Long = fact[n]

    // (n!)^{-1}
    fun invFact(n: Int): Long = invFact[n]

    // nCk
    fun nCr(n: Int, r: Int): Long {
        if (r < 0 || r > n) return 0
        return (((fact[n] * invFact[r]) % mod) * invFact[n - r]) % mod
    }

    // nPk = n! / (n-k)!
    fun nPr(n: Int, r: Int): Long {
        if (r < 0 || r > n) return 0
        return (fact[n] * invFact[n - r]) % mod
    }
}

fun minMaxSums(nums: IntArray, k: Int): Int {
    val mod = 1_000_000_007L
    val n = nums.size
    val combinators = Combinatorics(n, mod)
    val prefix = LongArray(n + 1)
    prefix[0] = 1L
    for (i in 1..n) {
        prefix[i] = prefix[i - 1] + combinators.nCr(n, i)
    }

    fun computeCSum(nMax: Int, kMax: Int): Long {
        if (kMax == 0) return 1L
        var result = 1L
        for (i in 1..kMax) {
            result = (result + combinators.nCr(nMax, i)) % mod
        }
        return result
    }

    nums.sort()
    var sum = 0L

    for (i in 0 until n) {
        val num = nums[i].toLong()
        val less = computeCSum(i, minOf(i, k - 1))
        val greater = computeCSum(n - i - 1, minOf(n - i - 1, k - 1))
       // println("$num $less $greater")

        val minSum = (less * num) % mod
        val maxSum = (greater * num) % mod

        sum = (sum + minSum) % mod
        sum = (sum + maxSum) % mod

    }
    return sum.toInt()
}

fun main() {
    println(
        minMaxSums(intArrayOf(1, 2, 3), 2)
    )
}