package contest

class Weekly473 {

    fun maxAlternatingSum(nums: IntArray): Long {
        val n = nums.size
        val k = n / 2
        val numbers = mutableListOf<Long>()
        for (num in nums) {
            numbers.add(num.toLong() * num.toLong())
        }
        numbers.sort()
        var ans = 0L
        for (i in 0 until n) {
            if (i < k) ans -= numbers[i] else ans += numbers[i]
        }
        return ans
    }

    fun countStableSubarrays(capacity: IntArray): Long {
        val n = capacity.size
        val groups = mutableMapOf<Long, MutableList<Int>>()
        val prefix = LongArray(n + 1)

        for (i in 0 until n) {
            val num = capacity[i].toLong()
            prefix[i + 1] = prefix[i] + num
            groups.computeIfAbsent(num) { mutableListOf() }.add(i)
        }

        var ans = 0L
        for ((num, list) in groups) {
            if (list.size < 2) continue
            val map = mutableMapOf<Long, Long>()
            var j = -2
            for (i in list) {
                val sum = prefix[i + 1]
                val prev = sum - 2L * num
                var cnt = (map[prev] ?: 0L)
                if (cnt > 0 && j == i - 1 && prefix[j + 1] == prev) cnt--
                ans += cnt
                map[sum] = (map[sum] ?: 0L) + 1L
                j = i
            }
        }
        return ans
    }
}