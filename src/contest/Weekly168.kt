package contest

class Weekly168 {

    fun lexSmallest(s: String): String {
        val n = s.length
        var ans = s
        for (k in 2..n) {
            val first = s.take(k).reversed() + s.takeLast(n - k)
            val last = s.take(n - k) + s.takeLast(k).reversed()
            ans = minOf(ans, first, last)
        }
        return ans
    }

    fun maxSumOfSquares(num: Int, sum: Int): String {
        val builder = StringBuilder()
        val q = sum / 9
        val r = sum % 9
        var cnt = q
        if (r != 0) cnt++
        if (cnt > num) return ""
        for (i in 0 until q) builder.append(9)
        if (r != 0) builder.append(r)
        for (i in cnt until num) builder.append(0)
        return builder.toString()
    }

}