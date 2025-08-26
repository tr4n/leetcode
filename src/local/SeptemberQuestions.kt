package local

fun areaOfMaxDiagonal(dimensions: Array<IntArray>): Int {
    val (width, length) = dimensions.maxWith(
        comparator = compareBy<IntArray> { it[0] * it[0] + it[1] * it[1] }
            .thenBy { it[0] * it[1] }
    )
    return width * length
}

fun sumOfFlooredPairs(nums: IntArray): Int {
    val mod = 1_000_000_007
    val n = nums.size
    var minValue = Int.MAX_VALUE
    var maxValue = Int.MIN_VALUE

    val distinctNumbers = mutableSetOf<Int>()
    for (num in nums) {
        distinctNumbers.add(num)
        minValue = minOf(minValue, num)
        maxValue = maxOf(maxValue, num)
    }

    val freq = IntArray(maxValue + 1)
    for (num in nums) freq[num]++
    val prefix = LongArray(maxValue + 1)
    for (i in 1..maxValue) prefix[i] = prefix[i - 1] + freq[i].toLong()

    var sum = 0L
    for (num in distinctNumbers) {
        val maxMultiple = maxValue / num

        // sum = (sum + num * freq[num]) % mod
        //   sum = (sum + prefix[minOf(2 * num - 1, maxValue)] - prefix[num]) % mod
        for (i in 1..maxMultiple) {
            val left = num * i
            val right = (num * (i + 1) - 1).coerceIn(left, maxValue)
            val total = (prefix[right] - prefix[left - 1]) * freq[num] * i
            //  println("$left $right ${freq[num]} $total")
            sum += total
            sum %= mod
        }
    }
    return (sum % mod).toInt()
}

fun minWastedSpace(packages: IntArray, boxes: Array<IntArray>): Int {
    val mod = 1_000_000_007
    var minValue = Int.MAX_VALUE
    var maxValue = Int.MIN_VALUE

    for (num in packages) {
        minValue = minOf(minValue, num)
        maxValue = maxOf(maxValue, num)
    }


    val freq = IntArray(maxValue + 1)
    for (num in packages) freq[num]++

    val prefixFreq = LongArray(maxValue + 1)
    val prefixSum = LongArray(maxValue + 1)
    for (i in minValue..maxValue){
        prefixFreq[i] = prefixFreq[i - 1] + freq[i].toLong()
        prefixSum[i] = prefixSum[i - 1] + freq[i].toLong() * i
    }

    boxes.onEach { it.sort() }
    val availableBoxes = boxes.filter { it.last() >= maxValue }

    var minWastedSpace = Long.MAX_VALUE
    for(boxList in availableBoxes) {
        var previousBox = 0
        var wasted = 0L
        for(box in boxList) {
            val current = box.coerceAtMost(maxValue)
            val previous = previousBox.coerceAtMost(maxValue)

            val count = prefixFreq[current] - prefixFreq[previous]
            val packageSum = prefixSum[current] - prefixSum[previous]
            val totalSpace = count * box
            wasted += (totalSpace - packageSum)
            previousBox = box
        //    println("$box $count $packageSum")
            if(wasted >= minWastedSpace) break
        }
        minWastedSpace = minOf(minWastedSpace, wasted)
    }
    return if(minWastedSpace == Long.MAX_VALUE) -1 else (minWastedSpace % mod).toInt()
}

fun main() {
    println(
        minWastedSpace(
            intArrayOf(3,5,8,10,11,12),
            "[[12],[11,9],[10,5,14]]".to2DIntArray()
        )
    )
}