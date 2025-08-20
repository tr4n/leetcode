import java.util.PriorityQueue
import java.util.TreeMap
import kotlin.collections.ArrayDeque
import kotlin.collections.sort
import kotlin.math.*

fun minCost(basket1: IntArray, basket2: IntArray): Long {
    val minValue = minOf(basket2.min(), basket1.min())
    val frequent1 = mutableMapOf<Int, Int>()
    val frequent2 = mutableMapOf<Int, Int>()

    for (fruit in basket1) {
        frequent1[fruit] = (frequent1[fruit] ?: 0) + 1
    }
    for (fruit in basket2) {
        frequent2[fruit] = (frequent2[fruit] ?: 0) + 1
    }

    // println(frequent1.toList())
    // println(frequent2.toList())

    val queue1 = ArrayDeque<Int>()
    val queue2 = ArrayDeque<Int>()

    for ((fruit, f1) in frequent1) {
        val f2 = frequent2[fruit] ?: 0
        if (f1 <= f2) continue
        if ((f1 + f2) % 2 != 0) return -1
        repeat((f1 - f2) / 2) { queue1.add(fruit) }

    }
    for ((fruit, f2) in frequent2) {
        val f1 = frequent1[fruit] ?: 0
        if (f2 <= f1) continue
        if ((f1 + f2) % 2 != 0) return -1
        repeat((f2 - f1) / 2) { queue2.add(fruit) }
    }

    queue1.sort()
    queue2.sort()


    //   println(queue1.toList())
    //  println(queue2.toList())
    if (queue1.size != queue2.size) return -1L
    val list = (queue1 + queue2).sorted()

    var cost = 0L
    for (i in 0 until (list.size / 2)) {
        cost += minOf(list[i], 2 * minValue)
    }
//
//    var sum1 = basket1.sum()
//    var sum2 = basket2.sum()
//    while (queue1.isNotEmpty() && queue2.isNotEmpty()) {
//        val min1 = queue1.first()
//        val max1 = queue1.last()
//        val min2 = queue2.first()
//        val max2 = queue2.last()
//
//
//        if (min1 + max2 < min2 + max1) {
//            queue1.removeFirst()
//            queue2.removeLast()
//            sum1 = sum1 - min1 + max2
//            sum2 = sum2 + min1 - max2
//            cost += minOf(min1, max2, 2 * minValue).toLong()
//        } else {
//            queue1.removeLast()
//            queue2.removeFirst()
//            sum1 = sum1 - max1 + min2
//            sum2 = sum2 + max1 - min2
//            cost += minOf(min2, max1, 2 * minValue).toLong()
//        }
//        println("$sum1 $sum2")
//        if (sum1 == sum2) return cost
//    }

//    if (queue1.isNotEmpty() || queue2.isNotEmpty()) {
//        return -1L
//    }
    return cost
}

fun integerReplacement(n: Int): Int {
    val queue = PriorityQueue<Pair<Long, Int>>(compareBy({ it.second }, { (num, step) ->
        val log2 = log2(num.toFloat()).toInt()
        step + minOf(abs(num - (1 shl log2)), abs(num - (1 shl (log2 + 1))))
    }))

    val map = mutableMapOf<Long, Int>()
    map[n.toLong()] = 0
    queue.add(n.toLong() to 0)

    while (queue.isNotEmpty()) {
        val (number, currentStep) = queue.poll()

        if (number == 1L) {
            return currentStep
        }
        val nextStep = currentStep + 1
        val nextNumbers = if (number % 2 == 0L) {
            listOf(number / 2L)
        } else {
            listOf(number - 1L, number + 1L)
        }

        for (next in nextNumbers) {
            val currentNextStep = map[next] ?: Int.MAX_VALUE
            if (nextStep < currentNextStep) {
                map[next] = nextStep
                queue.add(next to nextStep)
            }
        }
    }
    return map[1] ?: Int.MAX_VALUE
}

fun maxSubArray(nums: IntArray): Int {
    val n = nums.size

    var currentMax = nums[0]
    var globalMax = nums[0]

    for (i in 1 until n) {
        currentMax = maxOf(nums[i], currentMax + nums[i])
        globalMax = maxOf(globalMax, currentMax)
    }
    return globalMax
}

fun maxProduct(nums: IntArray): Int {
    val n = nums.size

    var currentMax = nums[0]
    var currentMin = nums[0]
    var globalMax = nums[0]

    for (i in 1 until n) {
        val productMax = currentMax * nums[i]
        val productMin = currentMin * nums[i]
        currentMax = nums[i]
        currentMin = nums[i]
        if (productMax >= 0) {
            currentMax = max(currentMax, productMax)
        } else {
            currentMin = min(currentMin, productMax)
        }
        if (productMin >= 0) {
            currentMax = max(currentMax, productMin)
        } else {
            currentMin = min(currentMin, productMin)
        }
        globalMax = maxOf(globalMax, currentMax)
    }
    return globalMax
}

fun subarraySum(nums: IntArray, k: Int): Int {
    val sumMap = mutableMapOf<Int, Int>()
    sumMap[0] = 1 // cần khởi tạo để tính subarray bắt đầu từ index 0

    var count = 0
    var prefixSum = 0

    for (num in nums) {
        prefixSum += num
        count += sumMap[prefixSum - k] ?: 0
        sumMap[prefixSum] = (sumMap[prefixSum] ?: 0) + 1
    }

    return count
}

fun checkSubarraySum(nums: IntArray, k: Int): Boolean {
    val sumMap = mutableMapOf<Int, Int>()
    sumMap[0] = 1

    var count = 0
    var prefixSum = 0

    for (num in nums) {
        prefixSum += num
        var modK = prefixSum % k
        modK = if (modK < 0) k + modK else modK
        count += sumMap[modK] ?: 0
        sumMap[modK] = (sumMap[modK] ?: 0) + 1
    }

    return count > 0
}

private fun findFirstIndex(list: List<Int>, k: Int): Int {
    var l = 0
    var r = list.size - 1
    var result = -1
    while (l <= r) {
        val mid = (l + r) / 2
        if (list[mid] < k) {
            l = mid + 1
        } else {
            if (list[mid] == k) result = mid
            r = mid - 1
        }
    }
    return result
}

private fun findLastIndex(list: List<Int>, k: Int): Int {
    var l = 0
    var r = list.size - 1
    var result = -1
    while (l <= r) {
        val mid = (l + r) / 2
        if (list[mid] > k) {
            r = mid - 1
        } else {
            if (list[mid] == k) result = mid
            l = mid + 1
        }
    }
    return result
}

fun sumOfNumberAndReverse(num: Int): Boolean {
    if (num == 0) return true
    if (num == 1) return false
    if (num < 20 && num % 2 == 0) return true
    if (num < 10) return false
    for (n in num / 2 until num) {
        val r = n.toString().reversed().toInt()
        if (n + r == num) return true
    }
    return false
}

fun minimumNumbers(num: Int, k: Int): Int {
    if (num == 0) return 0
    if (k == 0) return if (num % 10 == 0) 1 else 0
    if (num % 10 == k) return 1

    for (i in 0..num / k) {
        if (num - k * i % 10 == 0) return i
    }
    return -1
}

fun pancakeSort(arr: IntArray): List<Int> {
    val n = arr.size

    var list = arr.toList()
    var max = n
    val answers = mutableListOf<Int>()
    while (max > 1) {
        //  println(list)
        for (i in 0 until n) {
            if (list[i] != max) continue
            if (i == n - 1) break
            answers.add(i + 1)
            answers.add(max)
            list = list.subList(0, i + 1).reversed() + list.subList(i + 1, n)
            // println(list)
            val first = list.subList(0, max).reversed()
            val second = if (max < n) list.subList(max, n) else emptyList()
            list = first + second
            break
        }
        max--
    }
    // println(list)
    return answers
}

fun isPowerOfTwo(n: Int): Boolean {
    return n > 0 && n and (n - 1) == 0
}


fun maximumGap(nums: IntArray): Int {
    val n = nums.size
    if (n < 2) return 0
    val max = nums.max()
    val min = nums.min()
    val bucketMin = IntArray(n) { max + 1 }
    val bucketMax = IntArray(n) { min - 1 }
    //   val buckets = Array(n) { PriorityQueue<Int>() }
    val bucketSize = ceil((max - min) / (n - 1).toDouble()).toInt().coerceAtLeast(1)

    for (i in 0 until n) {
        val num = nums[i]
        val k = (num - min) / bucketSize
        //      buckets[k].add(num)
        bucketMin[k] = minOf(bucketMin[k], num)
        bucketMax[k] = maxOf(bucketMax[k], num)
    }


    var gap = 0
    //   val mins = bucketMin.filter { it in min..max }
//    val maxes = bucketMax.filter { it in min..max }
    //  println(buckets.map { it.toList() })
//    for (i in 0 until n) {
//        val bucket = buckets[i]
//        while (bucket.size >= 2) {
//            gap = maxOf(gap, abs(bucket.poll() - bucket.peek()))
//        }
//    }

    var preMax = bucketMax[0]
    for (i in 1 until n) {
        val currentMin = bucketMin[i]
        if (currentMin !in min..max) continue
        gap = maxOf(gap, currentMin - preMax)
        preMax = bucketMax[i]
    }

    return gap
}

fun isPowerOfThree(n: Int): Boolean {
    if (n <= 0) return false
    val log3 = log(n.toFloat(), 3f)
    // println(log3)
    return 3.0.pow(log3.roundToInt()).toLong() == n.toLong()
}

fun miceAndCheese(reward1: IntArray, reward2: IntArray, k: Int): Int {
    val firstSum = reward1.sum()
    val secondSum = reward2.sum()
    val firstRewards = reward1.mapIndexed { index, value ->
        value - reward2[index]
    }.sortedByDescending { it }.take(k).sum()

    val secondRewards = reward2.mapIndexed { index, value ->
        value - reward1[index]
    }.sortedByDescending { it }.take(k).sum()

    val max1 = secondSum + firstRewards
    val max2 = firstSum + secondRewards
    if (max2 == secondSum && k == reward1.size) return max1
    //  println(max1)
    //   println(max2)
    return max1
}

fun minTimeToReach(moveTime: Array<IntArray>): Int {
    val maxTime = 1_500_000_003
    val m = moveTime.size
    val n = moveTime[0].size

    val directX = intArrayOf(0, 0, 1, -1)
    val directY = intArrayOf(1, -1, 0, 0)

    val d = Array(m) { IntArray(n) { maxTime } }
    d[0][0] = 0

    val queue = PriorityQueue<Pair<Int, Int>>(compareBy { (x, y) -> d[x][y] })
    queue.add(0 to 0)
    while (queue.isNotEmpty()) {
        val (cellX, cellY) = queue.poll()
        val waitingTimeSoFar = d[cellX][cellY]

        if (cellX == m - 1 && cellY == n - 1) {
            return waitingTimeSoFar
        }
        val step = 1 + (cellX + cellY) % 2
        for (i in 0 until 4) {
            val x = cellX + directX[i]
            val y = cellY + directY[i]
            if ((x !in 0 until m) || (y !in 0 until n)) continue
            val waitingTime = maxOf(waitingTimeSoFar, moveTime[x][y]) + step
            val currentTime = d[x][y]
            if (waitingTime < currentTime) {
                d[x][y] = waitingTime
                queue.add(x to y)
            }
        }
    }
    return 0
}

fun reorderedPowerOf2(n: Int): Boolean {
    if (n <= 20) return n in intArrayOf(
        1, 2, 4, 8, 16
    )

    val digits = n.toString().map { it.digitToInt() }
    val used = BooleanArray(digits.size)
    var number = 0
    var isPowerOf2 = false

    fun backtrack(pos: Int) {
        if (isPowerOf2) return
        if (pos == digits.size) {
            if (number > 0 && (number and (number - 1) == 0)) {
                isPowerOf2 = true
            }
            return
        }

        for (i in digits.indices) {
            if (used[i]) continue

            // Chặn ngay:
            if (pos == 0 && digits[i] == 0) continue
            if (pos == digits.size - 1 && (digits[i] % 2 != 0)) continue

            used[i] = true
            number = number * 10 + digits[i]
            backtrack(pos + 1)
            number /= 10
            used[i] = false
        }
    }
    backtrack(0)
    return isPowerOf2
}

fun maxArea(coords: Array<IntArray>): Long {
    val n = coords.size
    if (n < 3) return -1L
    val treeX = TreeMap<Int, Pair<Int, Int>>()
    val treeY = TreeMap<Int, Pair<Int, Int>>()

    for ((x, y) in coords) {
        val entryX = treeX[x]
        treeX[x] = if (entryX == null) {
            y to y
        } else {
            minOf(entryX.first, y) to maxOf(entryX.second, y)
        }

        val entryY = treeY[y]
        treeX[y] = if (entryY == null) {
            x to x
        } else {
            minOf(entryY.first, x) to maxOf(entryY.second, x)
        }
    }

    var area = 0L
    val firstX = treeX.firstEntry()?.key
    val lastX = treeX.lastEntry()?.key
    for (entry in treeX) {
        val x = entry.key
        val (yMin, yMax) = entry.value
        if (yMin == yMax) continue
        val baseY = abs(yMax - yMin).toLong()
        val triangleArea1 = if (firstX != null) abs(firstX - x).toLong() * baseY else 0L
        val triangleArea2 = if (lastX != null) abs(lastX - x).toLong() * baseY else 0L
        area = maxOf(area, triangleArea1, triangleArea2)
    }

    val firstY = treeY.firstEntry()?.key
    val lastY = treeY.lastEntry()?.key
    for (entry in treeY) {
        val y = entry.key
        val (xMin, xMax) = entry.value
        if (xMin == xMax) continue
        val baseX = abs(xMax - xMin).toLong()
        val triangleArea1 = if (firstY != null) abs(firstY - y).toLong() * baseX else 0L
        val triangleArea2 = if (lastY != null) abs(lastY - y).toLong() * baseX else 0L
        area = maxOf(area, triangleArea1, triangleArea2)
    }
    return if (area == 0L) -1 else area
}

fun resultsArray(nums: IntArray, k: Int): IntArray {
    val n = nums.size
    if (k == 1) return nums
    val result = IntArray(n - k + 1) { -1 }
    var left = 0
    var right = 1

    while (right < n && left <= n - k) {

        if (nums[right] <= nums[right - 1] || nums[right] - nums[right - 1] != -1) {
            left = right
            right++
            continue
        }
        if (right - left + 1 == k) {
            result[left] = nums[right]
            left++
            right++
            continue
        }
        right++
    }
    return result
}

fun findMaxSum(nums1: IntArray, nums2: IntArray, k: Int): LongArray {
    val n = nums1.size
    val pq = PriorityQueue<Long>()
    val numbers = nums1.withIndex()
        .sortedBy { it.value }
        .map { it.index to nums2[it.index].toLong() }

    val answers = LongArray(n) { -1L }
//    println(nums1.withIndex().map { it.index to it.value }.sortedBy { it.second })
//    println(numbers)
    //   val answerList = mutableListOf<Long>()

    var sum = 0L
    var preIndex = -1
    for ((i, num) in numbers) {
        answers[i] = if (preIndex >= 0 && nums1[i] == nums1[preIndex]) {
            answers[preIndex]
        } else sum
        //   answerList.add(answers[i])
        pq.add(num)
        sum += num
        if (pq.size > k) {
            sum -= pq.poll()
        }
        preIndex = i
    }
    //  println(answerList)
    return answers
}

fun maxKelements(nums: IntArray, k: Int): Long {
    val priorityQueue = PriorityQueue<Long>(compareByDescending { it })

    for (num in nums) {
        priorityQueue.add(num.toLong())
    }

    var score = 0L
    repeat(k) {
        val top = priorityQueue.poll()
        score += top
        priorityQueue.add(ceil(top.toDouble() / 3.0).toLong())
    }
    return score
}

fun minStoneSum(piles: IntArray, k: Int): Int {
    val priorityQueue = PriorityQueue<Int>(compareByDescending { it })

    for (num in piles) {
        priorityQueue.add(num)
    }

    repeat(k) {
        val top = priorityQueue.poll()
        priorityQueue.add(top - floor(0.5 * top.toDouble()).toInt())
    }
    return priorityQueue.sum()
}

fun halveArray(nums: IntArray): Int {
    val numbers = nums.map { it.toDouble() }
    val totalSum = numbers.sum()
    val priorityQueue = PriorityQueue<Double>(compareByDescending { it })

    for (num in numbers) {
        priorityQueue.add(num)
    }
    var reduceSum = 0.0
    var count = 0
    while (2 * reduceSum < totalSum) {
        val top = priorityQueue.poll()
        val reduce = 0.5 * top.toDouble()
        reduceSum += reduce
        count++
        priorityQueue.add(top - reduce)
    }
    return count
}

fun minOperations(nums: IntArray, k: Int): Int {
    val priorityQueue = PriorityQueue<Long>()

    for (num in nums) {
        priorityQueue.add(num.toLong())
    }

    var cnt = 0
    if (priorityQueue.peek() >= k) {
        return 0
    }
    while (priorityQueue.size >= 2) {
        val min = priorityQueue.poll()
        val max = priorityQueue.poll()
        val n = 2L * min + max
        priorityQueue.add(n)
        cnt++
        if (priorityQueue.peek() >= k) {
            return cnt
        }
    }
    return cnt
}

fun findUnsortedSubarray(nums: IntArray): Int {
    val n = nums.size
    var left = 1
    while (left < n && nums[left] >= nums[left - 1]) left++

    var right = n - 2
    while (right >= 0 && nums[right] <= nums[right + 1]) right--
    if (left >= n || right < 0) return 0

    var start = left - 1
    var end = right + 1
    if (start > end) {
        val tmp = start
        start = end
        end = tmp
    }

    var min = nums[start]
    var max = nums[start]
    for (i in (start + 1)..end) {
        min = minOf(min, nums[i])
        max = maxOf(max, nums[i])
    }

    start--
    while (start >= 0 && nums[start] > min) start--
    end++
    while (end < n && nums[end] < max) end++

    return end - start - 1
}

fun circularPermutation(n: Int, start: Int): List<Int> {
    val size = 1 shl n
    var startIndex = 0

    for (i in 0 until size) {
        if ((i xor (i shr 1)) == start) {
            startIndex = i
            break
        }
    }

    return List(size) { i ->
        val k = (i + startIndex) % size
        k xor (k shr 1)
    }
}

fun kthSmallest(matrix: Array<IntArray>, k: Int): Int {
    val m = matrix.size
    val n = matrix[0].size

    fun countLessThan(x: Int): Int {
        var cnt = 0
        for (row in matrix) {
            var l = 0
            var r = n - 1
            var result = -1
            while (l <= r) {
                val mid = l + (r - l) / 2
                val num = row[mid]
                if (num <= x) {
                    result = mid
                    l = mid + 1
                } else {
                    r = mid - 1
                }
            }
            cnt += (result + 1)
        }
        return cnt
    }
    //   println(matrix.joinToString("\n") { it.toList().toString() })

    var low = matrix[0][0]
    var high = matrix[m - 1][n - 1]
    while (low < high) {
        val mid = low + (high - low) / 2
        val count = countLessThan(mid)
        //  println("$mid $count")
        if (count < k) {
            //  if (count == k) result = mid
            low = mid + 1
        } else {
            high = mid
        }
    }
    return low
}

fun findKthNumber(m: Int, n: Int, k: Int): Int {

    fun countLessThan(x: Int): Int {
        var cnt = 0
        for (row in 0 until m) {
            var l = 0
            var r = n - 1
            var result = -1
            while (l <= r) {
                val mid = l + (r - l) / 2
                val num = (row + 1) * (mid + 1)
                if (num <= x) {
                    result = mid
                    l = mid + 1
                } else {
                    r = mid - 1
                }
            }
            cnt += (result + 1)
        }
        return cnt
    }
    //   println(matrix.joinToString("\n") { it.toList().toString() })
//    println(
//        List(m) {row ->
//            List(n) {col ->
//                (row + 1) * (col + 1)
//            }
//        }.joinToString("\n") { it.toString() }
//    )
    var low = 1
    var high = m * n
    while (low < high) {
        val mid = low + (high - low) / 2
        val count = countLessThan(mid)
        //   println("$mid $count")
        if (count < k) {
            //  if (count == k) result = mid
            low = mid + 1
        } else {
            high = mid
        }
    }
    return low
}

fun kSmallestPairs(nums1: IntArray, nums2: IntArray, k: Int): List<List<Int>> {

    val count1 = mutableMapOf<Int, Int>()
    val count2 = mutableMapOf<Int, Int>()

    for (num in nums1) {
        count1[num] = (count1[num] ?: 0) + 1
    }

    for (num in nums2) {
        count2[num] = (count2[num] ?: 0) + 1
    }

    val numbers1 = nums1.distinct()
    val numbers2 = nums2.distinct()
    val m = count1.size
    val n = count2.size
    //  println(numbers1)
    //   println(numbers2)
    val pq = PriorityQueue<List<Int>>(compareBy { (i, j) ->
        numbers1[i] + numbers2[j]
    })

    val set = mutableSetOf<List<Int>>()

    for (j in 0 until n) {
        val element = listOf(0, j)
        set.add(element)
        pq.add(element)
    }
    val result = mutableListOf<List<Int>>()
    var count = k
    while (count > 0) {
        val (i, j) = pq.poll()
        val a = numbers1[i]
        val b = numbers2[j]
        var pairCount = (count1[a] ?: 0) * (count2[b] ?: 0)
        while (count > 0 && pairCount > 0) {
            result.add(listOf(a, b))
            count--
            pairCount--
        }
        if (count == 0) {
            return result
        }

        //    println("${count + 1}: $num " + pq.map { it.first() })
        val next1 = listOf(i + 1, j)
        if (i + 1 < m && next1 !in set) {
            pq.add(next1)
            set.add(next1)
        }

        val next2 = listOf(i, j + 1)
        if (j + 1 < n && next2 !in set) {
            pq.add(next2)
            set.add(next2)
        }
    }
    return result
}

fun findClosestElements(arr: IntArray, k: Int, x: Int): List<Int> {
    val n = arr.size
    var idx = arr.binarySearch(x)
    if (idx < 0) {
        idx = intArrayOf(-idx - 1, -idx - 2, -idx).minBy {
            if (it !in 0 until n) Int.MAX_VALUE else abs(arr[it] - x)
        }
    }
    println("n = $n, k = $k, x= $x, idx = $idx, ")
    val pq = PriorityQueue<Int>(
        compareByDescending<Int> { abs(it - x) }
            .thenBy { -it }
    )
    pq.add(arr[idx])
    var left = idx - 1
    var right = idx + 1
    val lowerIndex = (idx - k - 1).coerceAtLeast(0)
    val upperIndex = (idx + k + 1).coerceAtMost(n - 1)
    var count = k

    while (count > 0 && (left >= lowerIndex || right <= upperIndex)) {
        if (left >= lowerIndex) {
            pq.add(arr[left])
            if (pq.size > k) pq.poll()
            left--
            count--
        }
        if (right <= upperIndex) {
            pq.add(arr[right])
            if (pq.size > k) pq.poll()
            right++
            count--
        }

    }
    return pq.sorted()
}

fun maximum69Number(num: Int): Int {
    val digits = num.toString().toCharArray()
    return (0 until digits.size).maxOf { idx ->
        digits.mapIndexed { i, c ->
            if (i == idx) '9' else c
        }.joinToString("").toInt()
    }
}

fun maximumScore(a: Int, b: Int, c: Int): Int {
    val x = minOf(a, b, c)
    val z = maxOf(a, b, c)
    val y = a + b + c - x - z

    return if (z > x + y) x + y else (x + y + z) / 2
}

fun minCost(colors: String, neededTime: IntArray): Int {
    val n = colors.length
    if (n == 1) return 0
    val totalTime = neededTime.sum()
    var remainingTime = 0
    var maxSoFar = neededTime[0]
    for (i in 1 until n) {
        if (colors[i] != colors[i - 1]) {
            remainingTime += maxSoFar
            maxSoFar = neededTime[i]
            continue
        }
        maxSoFar = maxOf(maxSoFar, neededTime[i])
    }
    remainingTime += maxSoFar
    return totalTime - remainingTime
}

fun findCheapestPrice(n: Int, flights: Array<IntArray>, src: Int, dst: Int, k: Int): Int {
    val maxValue = 1_000_003

    var dist = IntArray(n) { maxValue }
    dist[src] = 0

    repeat(k + 1) {
        val next = dist.clone()
        for ((u, v, cost) in flights) {
            if (dist[u] != maxValue && dist[u] + cost < next[v]) {
                next[v] = dist[u] + cost
            }
        }
        dist = next
    }

    return if (dist[dst] >= maxValue) -1 else dist[dst]
}

fun topKFrequent(words: Array<String>, k: Int): List<String> {
    val frequencies = words.groupingBy { it }.eachCount()
    return frequencies.keys.sortedWith(
        compareByDescending<String> { frequencies[it] ?: 0 }
            .thenBy { it }
    ).take(k)
}

fun largestWordCount(messages: Array<String>, senders: Array<String>): String {
    val map = mutableMapOf<String, Int>()
    for (i in 0 until messages.size) {
        val wordCount = messages[i].count { it == ' ' } + 1
        val sender = senders[i]
        map[sender] = (map[sender] ?: 0) + wordCount
    }
    return map.keys.maxWithOrNull(
        compareBy<String> { map[it] ?: 0 }
            .thenBy { it }
    ) ?: ""
}

fun countSubarrays(nums: IntArray, k: Int): Int {
    val n = nums.size
    val counts = IntArray(n)
    val balances = IntArray(n)
    counts[0] = if (nums[0] == k) 1 else 0
    balances[0] = when {
        nums[0] < k -> -1
        nums[0] > k -> 1
        else -> 0
    }

    for (i in 1 until n) {
        val num = nums[i]
        val x = when {
            num < k -> -1
            num > k -> 1
            else -> 0
        }
        val y = 1 - abs(x)
        balances[i] = balances[i - 1] + x
        counts[i] = counts[i - 1] + y

    }

    fun countLessThan(list: List<Int>?, x: Int): Int {
        if (list.isNullOrEmpty()) return 0
        var l = 0
        var r = list.size
        while (l < r) {
            val mid = (l + r) / 2
            if (counts[list[mid]] < counts[x]) l = mid + 1 else r = mid
        }
        return l
    }

    //  println(balances.toList())
    // println(counts.toList())
    val indexes = mutableMapOf<Int, MutableList<Int>>()
    var cnt = 0

    for (i in 0 until n) {
        if (counts[i] > 0 && (balances[i] == 0 || balances[i] == 1)) cnt++
    }

    for (i in 0 until n) {
        val balance = balances[i]
        val odd = countLessThan(indexes[balance], i)
        val even = countLessThan(indexes[balance - 1], i)
        // println("$balance $odd $even")
        cnt += (odd + even)
        if (indexes[balance] == null) {
            indexes[balance] = mutableListOf(i)
        } else indexes[balance]?.add(i)
    }
    return cnt
}

fun new21Game(n: Int, k: Int, maxPts: Int): Double {
    val m = k + maxPts
    if (n < k) return 0.0
    if (n >= m) return 1.0
    if (k == 0) return 1.0

    val dp = DoubleArray(m)
    val sums = DoubleArray(m)
    dp[0] = 1.0
    sums[0] = 1.0

    for (i in 1 until m) {
        val start = (i - maxPts).coerceAtLeast(0)
        val end = (i - 1).coerceAtMost(k - 1)
        //  println("$i  $start $end")
        val preSum = if (start <= 0) 0.0 else sums[start - 1]
        val totalSum = sums[end]
        val sum = totalSum - preSum
        dp[i] = sum / maxPts
        sums[i] = sums[i - 1] + dp[i]
    }

    // println(dp.toList())
    return sums[n] - sums[k - 1]
}

fun main() {
    // val matrix = "[[1,5,9],[10,11,13],[12,13,15]]".to2DIntArray()
    val flights = "[[0,1,2],[1,2,1],[2,0,10]]".to2DIntArray()
    println(
        new21Game(21, 17, 10)
    )
}