package home

import java.util.*
import kotlin.text.iterator

class MinimumWindowSubString(private val data: IntArray) {
    private val n = data.size
    private val tree = IntArray(4 * n)

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = data[l]
        } else {
            val mid = (l + r) / 2
            build(node * 2, l, mid)
            build(node * 2 + 1, mid + 1, r)
            tree[node] = maxOf(tree[node * 2], tree[node * 2 + 1])
        }
    }

    fun getMax(): Int {
        return query(1, 0, n - 1, 0, n - 1)
    }

    fun get(idx: Int): Int {
        return query(1, 0, n - 1, idx, idx)
    }

    private fun query(node: Int, l: Int, r: Int, i: Int, j: Int): Int {
        if (r < i || l > j) return Int.MIN_VALUE

        if (i <= l && r <= j) return tree[node]

        val mid = (l + r) / 2
        val left = query(node * 2, l, mid, i, j)
        val right = query(node * 2 + 1, mid + 1, r, i, j)
        return maxOf(left, right)
    }

    fun update(idx: Int, value: Int) {
        update(1, 0, n - 1, idx, value)
    }

    private fun update(node: Int, l: Int, r: Int, idx: Int, value: Int) {
        if (l == r) {
            tree[node] = value
        } else {
            val mid = (l + r) / 2
            if (idx <= mid) {
                update(node * 2, l, mid, idx, value)
            } else {
                update(node * 2 + 1, mid + 1, r, idx, value)
            }
            tree[node] = maxOf(tree[node * 2], tree[node * 2 + 1])
        }
    }
}

fun minWindow(s: String, t: String): String {
    val m = s.length
    val n = t.length

    val counts = IntArray(123)
    for (c in t) {
        counts[c.code]++
    }
    val tree = MinimumWindowSubString(counts)

    val tSet = t.toHashSet()

    var left = 0
    var right = 0
    //   println(s)
    //   println(t)
    val results = PriorityQueue<Pair<Int, Int>>(compareBy { it.second - it.first })

    while (right < m && left <= m - n) {
        while (left <= m - n && s[left] !in tSet) left++
        if (right < left) right = left
        if(right >= m) break
        val c = s[right]
        val count = counts[c.code]
        counts[c.code] = count - 1
        tree.update(c.code, count - 1)
        while (tree.getMax() <= 0) {
            results.add(left to right)
            val firstChar = s[left]
            counts[firstChar.code]++
            tree.update(firstChar.code, counts[firstChar.code])
            left++
        }
        right++
    }

   //   println(left to right)
  //    println(s.substring(left, right))

    while (right >= m && left <= m - n && tree.getMax() <= 0) {
        results.add(left to m - 1)
        val firstChar = s[left]
        counts[firstChar.code]++
        tree.update(firstChar.code, counts[firstChar.code])
        left++
    }
//    println(results.map {
//        s.substring(it.first, it.second + 1)
//    })
    if (results.isEmpty()) return ""
    val (l, r) = results.poll()
    return s.substring(l, r + 1)
}


fun main() {
    println(
        minWindow(
            "bbaac",
            "aba"
        )
    )
}