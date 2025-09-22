package local

import java.util.TreeMap

class Router(private val memoryLimit: Int) {
    data class Pack(val source: Int, val destination: Int, val timestamp: Int)

    private val set = mutableSetOf<Pack>()
    private val queue = ArrayDeque<Pack>()

    private val timestamps = mutableMapOf<Int, MutableList<Int>>()
    private val fenwicks = mutableMapOf<Int, Fenwick>()

    fun addPacket(source: Int, destination: Int, timestamp: Int): Boolean {
        val pack = Pack(source, destination, timestamp)
        if (!set.add(pack)) return false

        if (queue.size == memoryLimit) {
            forwardPacket()
        }
        queue.addLast(pack)

        val tsList = timestamps.computeIfAbsent(destination) { mutableListOf() }
        tsList.add(timestamp)

        val comp = tsList.distinct().sorted()
        val index = comp.binarySearch(timestamp).let { if (it < 0) -it - 1 else it }
        val fenwick = fenwicks.getOrPut(destination) { Fenwick(comp.size) }
        fenwick.update(index + 1, 1)

        return true
    }

    fun forwardPacket(): IntArray {
        if (queue.isEmpty()) return intArrayOf()
        val oldest = queue.removeFirst()
        set.remove(oldest)

        val comp = timestamps[oldest.destination]!!.distinct().sorted()
        val index = comp.binarySearch(oldest.timestamp)
        fenwicks[oldest.destination]!!.update(index + 1, -1)

        return intArrayOf(oldest.source, oldest.destination, oldest.timestamp)
    }

    fun getCount(destination: Int, startTime: Int, endTime: Int): Int {
        val comp = timestamps[destination]?.distinct()?.sorted() ?: return 0
        val fenwick = fenwicks[destination] ?: return 0

        val l = comp.binarySearch(startTime).let { if (it < 0) -it - 1 else it }
        val r = comp.binarySearch(endTime).let { if (it < 0) -it - 2 else it }

        if (l > r) return 0
        return fenwick.query(r + 1) - fenwick.query(l)
    }

    class Fenwick(val n: Int) {
        private val bit = IntArray(n + 1)
        fun update(i: Int, delta: Int) {
            var x = i
            while (x <= n) {
                bit[x] += delta
                x += x and -x
            }
        }
        fun query(i: Int): Int {
            var x = i
            var res = 0
            while (x > 0) {
                res += bit[x]
                x -= x and -x
            }
            return res
        }
    }
}
