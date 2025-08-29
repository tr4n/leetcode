package local

import java.util.LinkedList

class FrontMiddleBackQueue {
    val list = LinkedList<Int>()

    fun pushFront(`val`: Int) {
        list.addFirst(`val`)
    }

    fun pushMiddle(`val`: Int) {
        list.add(list.size / 2, `val`)
    }

    fun pushBack(`val`: Int) {
        list.addLast(`val`)
    }

    fun popFront(): Int {
        return list.removeFirstOrNull() ?: -1
    }

    fun popMiddle(): Int {
        val size = list.size
        return when {
            size == 0 -> -1
            size == 1 -> list.removeFirst()
            size % 2 == 0 -> list.removeAt(size / 2 - 1)
            else -> list.removeAt(size / 2)
        }
    }

    fun popBack(): Int {
        return list.removeLastOrNull() ?: -1
    }

}