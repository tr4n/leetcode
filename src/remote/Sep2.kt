package remote

import topic.TreeNode
import java.util.*
import kotlin.math.abs

fun displayTable(orders: List<List<String>>): List<List<String>> {
    val tables = mutableSetOf<String>()
    val foods = mutableSetOf<String>()
    val map = mutableMapOf<String, MutableMap<String, Int>>()
    //  val sum = mutableMapOf<String, Int>()

    for ((_, table, food) in orders) {
        tables.add(table)
        foods.add(food)
        val quantity = map.computeIfAbsent(table) { mutableMapOf() }[food] ?: 0
        map[table]?.set(food, quantity + 1)
//        sum[table] = (sum[table] ?: 0) + 1
    }


    val foodList = foods.sorted()
    val tableList = tables.sortedBy { it.toInt() }

    val result = mutableListOf<MutableList<String>>()
    result.add(mutableListOf())
    result[0].add("Table")
    result[0].addAll(foodList)

    for (table in tableList) {
        val list = mutableListOf<String>()
        list.add(table)
        for (food in foodList) {
            list.add((map[table]?.get(food) ?: 0).toString())
        }
        result.add(list)
    }

    return result
}

fun findBottomLeftValue(root: TreeNode?): Int {
    var depthMost = -1
    var leftMost = root?.`val` ?: return -1

    fun dfs(node: TreeNode?, depth: Int) {
        node ?: return
        if (node.left == null && depth > depthMost) {
            depthMost = depth
            leftMost = node.`val`
        }
        if (node.left != null) dfs(node.left, depth + 1)
        if (node.right != null) dfs(node.right, depth + 1)
    }
    dfs(root, 0)
    return leftMost
}

fun candy(ratings: IntArray): Int {
    val n = ratings.size
    var min = 0
    var num = 0
    var total = 0
    var sumSoFar = 0
    var cnt = 1

    for (i in 1 until n) {
        when {
            ratings[i] == ratings[i - 1] -> {
                sumSoFar += (1 - min) * cnt
                total += sumSoFar
                cnt = 1
                sumSoFar = 0
                num = 0
                continue
            }

            ratings[i] > ratings[i - 1] -> num++

            ratings[i] < ratings[i - 1] -> num--
        }
        cnt++
        sumSoFar += num
        min = minOf(min, num)
    }
    println(min)
    return total
}

fun minDiffInBST(root: TreeNode?): Int {
    val heap = PriorityQueue<Int>()

    fun dfs(node: TreeNode?) {
        node ?: return
        heap.add(node.`val`)
        dfs(node.left)
        dfs(node.right)
    }
    dfs(root)
    var minDist = Int.MAX_VALUE
    var prev = heap.poll()
    while (heap.isNotEmpty()) {
        minDist = minOf(minDist, abs(heap.peek() - prev))
        if(minDist == 0) return 0
        prev = heap.poll()
    }

    return minDist
}


fun compareVersion(version1: String, version2: String): Int {
    val revision1 = version1.split(".")
    val revision2 = version2.split(".")
    val n = maxOf(revision1.size, revision2.size)

    for (i in 0 until n) {
        val a = revision1.getOrNull(i)?.toIntOrNull() ?: 0
        val b = revision2.getOrNull(i)?.toIntOrNull() ?: 0
        if (a == b) continue
        return if (a > b) 1 else -1
    }
    return 0
}

fun main() {
    println(
        candy(intArrayOf(6, 7, 6, 5, 4, 3, 2, 1, 0, 0, 0, 1, 0))
    )
}