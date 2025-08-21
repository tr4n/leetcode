package remote

import java.util.*

class MyCalendar {
    private val treeMap = TreeMap<Int, Int>()

    fun book(start: Int, end: Int): Boolean {

        val floor = treeMap.floorEntry(start)
        if (floor != null && floor.value > start) return false

        val ceiling = treeMap.ceilingEntry(start)
        if (ceiling != null && ceiling.key < end) return false

        treeMap[start] = end
        return true
    }
}


class MyCalendar2 {
    private val treeMap = TreeMap<Int, Int>()

    fun book(start: Int, end: Int): Boolean {
        treeMap[start] = treeMap.getOrDefault(start, 0) + 1
        treeMap[end] = treeMap.getOrDefault(end, 0) - 1
        var isAbleToBook = true
        val counts = TreeMap<Int, Int>()
        var count = 0
        for ((time, delta) in treeMap) {
            count += delta
            counts[time] = count
            if (time > end) break
            if (time in start until end) {
                if (count > 2) {
                    isAbleToBook = false
                    break
                }
            }
        }
        println("[$start,$end) (${treeMap[start]}, ${treeMap[end - 1]}) $isAbleToBook")
        println(counts.entries)
        if (!isAbleToBook) {
            treeMap[start] = treeMap.getOrDefault(start, 0) - 1
            treeMap[end] = treeMap.getOrDefault(end, 0) + 1
        }

        return isAbleToBook
    }
}


class MyCalendar3 {
    private val treeMap = TreeMap<Int, Int>()

    fun book(start: Int, end: Int): Boolean {
        treeMap[start] = treeMap.getOrDefault(start, 0) + 1
        treeMap[end] = treeMap.getOrDefault(end, 0) - 1
        var isAbleToBook = true
        val counts = TreeMap<Int, Int>()
        var count = 0
        for ((time, delta) in treeMap) {
            count += delta
            counts[time] = count
            if (time > end) break
            if (time in start until end) {
                if (count > 2) {
                    isAbleToBook = false
                }
            }
        }
        //  println("[$start,$end) (${treeMap[start]}, ${treeMap[end - 1]}) $isAbleToBook")
        //   println(counts.entries)
        if (!isAbleToBook) {
            treeMap[start] = treeMap.getOrDefault(start, 0) - 1
            treeMap[end] = treeMap.getOrDefault(end, 0) + 1
        }

        return isAbleToBook
    }
}


fun carPooling(trips: Array<IntArray>, capacity: Int): Boolean {
    val treeMap = TreeMap<Int, Int>()

    for ((numPassengers, from, to) in trips) {
        treeMap[from] = treeMap.getOrDefault(from, 0) + numPassengers
        treeMap[to] = treeMap.getOrDefault(to, 0) - numPassengers
    }
    var count = 0
    for ((time, delta) in treeMap) {
        count += delta
        if (count < 0) count = 0
        if (count > capacity) return false
    }
    return true
}
