package local

class LazyMinWithShiftedIndexSegmentTree(private val data: IntArray) {
    data class Node(
        var value: Int = Int.MAX_VALUE,
        var index: Int = -1,
        var rawIndex: Int = -1
    )

    private val n = data.size
    private val tree = Array(4 * n) { Node() }
    private val lazyShift = IntArray(4 * n)

    init {
        build(1, 0, n - 1)
    }

    private fun build(node: Int, l: Int, r: Int) {
        if (l == r) {
            tree[node] = Node(data[l], l, l)
            return
        }
        val mid = (l + r) / 2
        build(node * 2, l, mid)
        build(node * 2 + 1, mid + 1, r)
        tree[node] = merge(tree[node * 2], tree[node * 2 + 1])
    }

    // SHIFT  [ql, qr]  delta
    private fun applyShift(node: Int, delta: Int) {
        tree[node].index += delta
        lazyShift[node] += delta
    }

    fun shift(ql: Int, qr: Int, delta: Int) {
        shift(1, 0, n - 1, ql, qr, delta)
    }

    private fun shift(node: Int, l: Int, r: Int, ql: Int, qr: Int, d: Int) {
        if (qr < l || r < ql) return
        if (ql <= l && r <= qr) {
            lazyShift[node] += d; return
        }
        val m = (l + r) / 2
        shift(node * 2, l, m, ql, qr, d)
        shift(node * 2 + 1, m + 1, r, ql, qr, d)
        tree[node] = merge(tree[node * 2], tree[node * 2 + 1]) // value/min theo rawIndex
    }

    private fun push(node: Int) {
        if (lazyShift[node] != 0) {
            applyShift(node * 2, lazyShift[node])
            applyShift(node * 2 + 1, lazyShift[node])
            lazyShift[node] = 0
        }
    }


    // UPDATE value at rawIndex
    fun updateValue(rawIndex: Int, newValue: Int) {
        updateValue(1, 0, n - 1, rawIndex, newValue)
    }

    private fun updateValue(node: Int, l: Int, r: Int, rawIndex: Int, newValue: Int) {
        if (l == r) {
            tree[node].value = newValue
            return
        }
        pushDown(node)
        val mid = (l + r) / 2
        if (rawIndex <= mid) updateValue(node * 2, l, mid, rawIndex, newValue)
        else updateValue(node * 2 + 1, mid + 1, r, rawIndex, newValue)
        tree[node] = merge(tree[node * 2], tree[node * 2 + 1])
    }

    fun queryMin(ql: Int, qr: Int): Node =
        queryMin(1, 0, n - 1, ql, qr, 0)

    private fun queryMin(
        node: Int, l: Int, r: Int, ql: Int, qr: Int, acc: Int
    ): Node {
        if (qr < l || r < ql) return Node(Int.MAX_VALUE, Int.MAX_VALUE, -1)
        val curAcc = acc + lazyShift[node]
        if (ql <= l && r <= qr) {
            val nd = tree[node]
            return Node(nd.value, nd.rawIndex + curAcc, nd.rawIndex)
        }
        val m = (l + r) / 2
        val L = queryMin(node * 2, l, m, ql, qr, curAcc)
        val R = queryMin(node * 2 + 1, m + 1, r, ql, qr, curAcc)
        return when {
            L.value < R.value -> L
            R.value < L.value -> R
            else -> if (L.rawIndex <= R.rawIndex) L else R // tie-break theo index hiện tại
        }
    }


    // QUERY currentList (O(n))
    fun queryList(): List<Node> {
        val out = mutableListOf<Node>()
        collect(1, 0, n - 1, 0, out)
       // out.sortBy { it.index } // sắp xếp theo index hiện tại
        return out
    }

    private fun collect(
        node: Int, l: Int, r: Int, acc: Int, out: MutableList<Node>
    ) {
        val curAcc = acc + lazyShift[node]
        if (l == r) {
            val nd = tree[node]
            out.add(Node(nd.value, nd.rawIndex + curAcc, nd.rawIndex))
            return
        }
        val m = (l + r) / 2
        collect(node * 2, l, m, curAcc, out)
        collect(node * 2 + 1, m + 1, r, curAcc, out)
    }

    private fun pushDown(node: Int) {
        val delta = lazyShift[node]
        if (delta != 0) {
            for (child in listOf(node * 2, node * 2 + 1)) {
                lazyShift[child] += delta
                tree[child].index += delta
            }
            lazyShift[node] = 0
        }
    }

    private fun merge(a: Node, b: Node): Node {
        return when {
            a.value < b.value -> a
            b.value < a.value -> b
            else -> if (a.index < b.index) a else b
        }
    }
}


fun minInteger(num: String, k: Int): String {
    val arr = num.map { it.digitToInt() }
    val tree = LazyMinWithShiftedIndexSegmentTree(arr.toIntArray())
    val sumTree = SumSegmentTree(IntArray(arr.size) {1})
    val minNum = arr.sorted().joinToString("")
    println(arr)
    val n = arr.size
    var t = k
    val result = mutableListOf<Int>()
    val indexSet = mutableSetOf<Int>()
    while (t > 0) {
        println("t = $t")
        val start = result.size
        if (start == n) break
        var l = 0
        var r = n
        val target = t.coerceAtMost(n)
        while(l <= r) {
            val mid = (l + r)/2
            val count = sumTree.sumRange(0, mid)
            if(count > target) {
                r = mid - 1
            } else {
                l = mid
            }
        }
        val end = l
//        println(tree.queryList().sortedBy { it.index }.map {
//            //  arr[it.rawIndex] }
//            it.index
//        })
        val (minValue, minIndex, rawIndex) = tree.queryMin(0, end)
        println("$minValue-$minIndex-$rawIndex")
        if (minValue > 9) {
            break
        }
        tree.updateValue(rawIndex, 99)
        if (rawIndex > start) {
       //     tree.shift(rawIndex, rawIndex, -100)
            tree.shift(0, rawIndex - 1, +1)
        }
        println(tree.queryList().sortedBy { it.index }.map {
            //  arr[it.rawIndex] }
            it.index
        })
        result.add(minValue)
        indexSet.add(rawIndex)
        t -= minIndex
  //      println("$minIndex-$minValue")
//        println(tree.queryList().sortedBy { it.index }.map {
//            //  arr[it.rawIndex] }
//            it.index
//        })

    }

    for (i in 0 until n) {
        if (i !in indexSet) result.add(arr[i])
    }

    return result.joinToString("")
}

fun main() {
    println(
        minInteger("4321", 4)
    )
}