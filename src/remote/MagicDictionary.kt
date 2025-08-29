package remote

class MagicDictionary() {
    private class Node {
        val children = mutableMapOf<Char, Node>()
        var isWord = false
    }

    private val root = Node()

    fun buildDict(dictionary: Array<String>) {
        for (word in dictionary) {
            var node = root
            for (c in word) {
                node = node.children.computeIfAbsent(c) { Node() }
            }
            node.isWord = true
        }
    }

    fun search(searchWord: String): Boolean {
        fun dfs(node: Node, index: Int, modified: Boolean): Boolean {
            if (index == searchWord.length) {
                return modified && node.isWord
            }

            val c = searchWord[index]
            for ((ch, nextNode) in node.children) {
                if (ch == c) {
                    if (dfs(nextNode, index + 1, modified)) return true
                } else if (!modified) {
                    if (dfs(nextNode, index + 1, true)) return true
                }
            }
            return false
        }
        return dfs(root, 0, false)
    }
}


fun main(){
    val dict = MagicDictionary()
    dict.buildDict(arrayOf("hello","hallo","leetcode"))
    println(
    dict.search("hello")
    )
}