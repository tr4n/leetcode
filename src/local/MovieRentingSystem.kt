package local

import java.util.TreeSet

class MovieRentingSystem(n: Int, entries: Array<IntArray>) {
    data class Entry(val shop: Int, val movie: Int, val price: Int)

    private val available = mutableMapOf<Int, TreeSet<Entry>>()
    private val allEntries = mutableMapOf<Pair<Int, Int>, Entry>()

    private val rentedSet = TreeSet<Entry>(
        compareBy<Entry> { it.price }
            .thenBy { it.shop }
            .thenBy { it.movie }
    )

    init {
        for ((shop, movie, price) in entries) {
            val entry = Entry(shop, movie, price)
            available.computeIfAbsent(movie) {
                TreeSet(compareBy<Entry> { it.price }.thenBy { it.shop })
            }.add(entry)
            allEntries[shop to movie] = entry
        }
    }

    fun search(movie: Int): List<Int> {
        val entries = available[movie] ?: return emptyList()
        var entry = entries.firstOrNull()
        val result = mutableListOf<Int>()
        while (entry != null && result.size < 5) {
            result.add(entry.shop)
            entry = entries.higher(entry)
        }
        return result
    }

    fun rent(shop: Int, movie: Int) {
        val entry = allEntries[shop to movie] ?: return
        available[movie]?.remove(entry)
        rentedSet.add(entry)
    }

    fun drop(shop: Int, movie: Int) {
        val entry = allEntries[shop to movie] ?: return
        rentedSet.remove(entry)
        available[movie]?.add(entry)
    }

    fun report(): List<List<Int>> {
        val result = mutableListOf<List<Int>>()
        var entry = rentedSet.firstOrNull()
        while (entry != null && result.size < 5) {
            result.add(listOf(entry.shop, entry.movie))
            entry = rentedSet.higher(entry)
        }
        return result
    }
}
