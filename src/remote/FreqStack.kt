package remote

import java.util.*

class FreqStack {
    private val map = mutableMapOf<Int, Int>()
    private val list = mutableListOf<Int>()
    private val tree = TreeMap<Int, MutableSet<Int>>()

    fun push(`val`: Int) {
        val freq = map[`val`] ?: 0
        map[`val`] = freq + 1

        if (tree[freq + 1] == null) {
            val set = TreeSet<Int>()
            set.add(`val`)
            tree[freq + 1] = set
        } else {
            tree[freq + 1]?.add(`val`)
        }

        if (freq > 0) {
            tree[freq]?.remove(`val`)
        }
        list.add(`val`)
    }

    fun pop(): Int {
        val mostFreq = runCatching { tree.lastKey() }.getOrNull() ?: return -1
        val set = tree[mostFreq]
        if (set.isNullOrEmpty()) {
            tree.remove(mostFreq)
            return pop()
        }
        //  println(tree)
        //  println(list)
        //   println(set.toList())

        val idx = list.indexOfLast { map[it] == mostFreq }
        if (idx < 0) return -1
        val num = list[idx]

        list.removeAt(idx)
        set.remove(num)

        if (set.isEmpty()) {
            tree.remove(mostFreq)
        } else {
            tree[mostFreq] = set
        }

        if (mostFreq > 0) tree[mostFreq - 1]?.add(num)


        if (mostFreq == 1) {
            map.remove(num)
        } else {
            map[num] = mostFreq - 1
        }

        //   println(num)

        return num
    }

}

fun main() {
    val freqStack = FreqStack()
    freqStack.push(5) // The stack is [5]
    freqStack.push(7) // The stack is [5,7]
    freqStack.push(5) // The stack is [5,7,5]
    freqStack.push(7) // The stack is [5,7,5,7]
    freqStack.push(4) // The stack is [5,7,5,7,4]
    freqStack.push(5) // The stack is [5,7,5,7,4,5]
    freqStack.pop() // return 5, as 5 is the most frequent. The stack becomes [5,7,5,7,4].
    freqStack.pop() // return 7, as 5 and 7 is the most frequent, but 7 is closest to the top. The stack becomes [5,7,5,4].
    freqStack.pop() // return 5, as 5 is the most frequent. The stack becomes [5,7,4].
    freqStack.pop() // return 4, as 4, 5 and 7 is the most frequent, but 4 is closest to the top. The stack becomes [5,7].

}