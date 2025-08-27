package remote

class StreamChecker(private val words: Array<String>) {
    class Node {
        val children = arrayOfNulls<Node>(26)
        var isWord = false
    }

    private val root = Node()
    private val stream = ArrayDeque<Char>()
    private var maxLength = 0

    init {
        for (word in words) {
            maxLength = maxOf(maxLength, word.length)
            var node = root
            for (i in word.length - 1 downTo 0) {
                val c = word[i] - 'a'
                if (node.children[c] == null) node.children[c] = Node()
                node = node.children[c]!!
            }
            node.isWord = true
        }
    }

    fun query(letter: Char): Boolean {
        stream.addLast(letter)
        if (stream.size > maxLength) stream.removeFirst()

        var node = root
        val list = stream.reversed()
        for (c in list) {
            node = node.children[c - 'a'] ?: return false
            if (node.isWord) return true
        }
        return false
    }

}