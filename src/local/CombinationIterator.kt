package local

class CombinationIterator(characters: String, combinationLength: Int) {
    private val n = characters.length
    private val k = combinationLength
    private val list = mutableListOf<String>()
    private var index = 0

    init {
        val total = 1 shl n

        for (mask in 0 until total) {
            val bitCount = Integer.bitCount(mask)
            if (bitCount != k) continue
            val sub = StringBuilder()
            for (i in 0 until n) {
                if (mask and (1 shl i) == 0) continue
                sub.append(characters[i])
            }
            list.add(sub.toString())
        }
        list.sort()
        println(list)
    }

    fun next(): String {
        return list.getOrNull(index++) ?: ""

    }

    fun hasNext(): Boolean {
        return index < list.size
    }

}

fun main() {
    val characters = "abcd"
    val combinationLength = 2
    val combinator = CombinationIterator(characters, combinationLength)
    combinator.next()
}