package topic

fun sumOfDistancesInTree(n: Int, edges: Array<IntArray>): IntArray {
    val graph = Array(n) { mutableListOf<Int>() }
    for ((u, v) in edges) {
        graph[u].add(v)
        graph[v].add(u)
    }

    val subTreeSize = IntArray(n) { 1 }
    val sumDistDown = IntArray(n)

    fun dfs(u: Int, parent: Int) {
        for (v in graph[u]) {
            if (v == parent) continue
            dfs(v, u)
            subTreeSize[u] += subTreeSize[v]
            sumDistDown[u] += sumDistDown[v] + subTreeSize[v]
        }
    }

    val result = IntArray(n)
    fun calc(u: Int, parent: Int) {
        for (v in graph[u]) {
            if (v == parent) continue
            result[v] = result[u] - 2 * subTreeSize[v] + n
            calc(v, u)
        }
    }
    dfs(0, -1)
    result[0] = sumDistDown[0]
    calc(0, -1)
    return result
}