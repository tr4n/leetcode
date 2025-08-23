package remote


fun xorQueries(arr: IntArray, queries: Array<IntArray>): IntArray {
    val prefix = IntArray(arr.size + 1)
    for (i in 0 until arr.size) {
        prefix[i + 1] = prefix[i] xor arr[i]
    }
    return IntArray(queries.size) {
        val (a, b) = queries[it]
        prefix[b + 1] xor prefix[a]
    }
}

fun main() {
    println(
        minimumDifference(intArrayOf(1), 10)
    )
}

