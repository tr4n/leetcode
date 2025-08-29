package local

import kotlin.text.iterator

class Trie {
    class TrieNode {
        val children = mutableMapOf<Char, TrieNode>()
        var isWord = false
    }

    val root = TrieNode()

    fun insert(word: String) {
        var node = root
        for (c in word) {
            node = node.children.computeIfAbsent(c) { TrieNode() }
        }
        node.isWord = true
    }

    fun search(word: String): Boolean {
        var node = root
        for (c in word) {
            node = node.children[c] ?: return false
        }
        return node.isWord
    }


    fun startsWith(prefix: String): Boolean {
        var node = root
        for (c in prefix) {
            node = node.children[c] ?: return false
        }
        return true
    }

    fun searchPrefix(word: String): String? {
        var node = root
        val result = StringBuilder()
        for (c in word) {
            node = node.children[c] ?: return null
            result.append(c)
            if (node.isWord) return result.toString()
        }
        return null
    }
}

fun replaceWords(dictionary: List<String>, sentence: String): String {
    val trie = Trie()
    for (word in dictionary) {
        trie.insert(word)
    }
    return sentence.split(" ").joinToString(" ") {
        trie.searchPrefix(it) ?: it
    }
}

fun findWords(board: Array<CharArray>, words: Array<String>): List<String> {
    val trie = Trie()
    for (word in words) trie.insert(word)
    val maxLength = words.maxOf { it.length }
    val m = board.size
    val n = board[0].size
    val result = mutableSetOf<String>()
    //  val visited = Array(m) { BooleanArray(n) }
    val dirX = intArrayOf(1, -1, 0, 0)
    val dirY = intArrayOf(0, 0, 1, -1)
    fun dfs(row: Int, col: Int, node: Trie.TrieNode, output: String, visited: Array<BooleanArray>) {
        visited[row][col] = true
        if (node.isWord) {
            result.add(output)
        }
        if (output.length >= maxLength) {
            visited[row][col] = false
            return
        }
        for (i in 0 until 4) {
            val x = row + dirX[i]
            val y = col + dirY[i]
            if (x !in 0 until m || y !in 0 until n) continue
            if (visited[x][y]) continue
            val ch = board[x][y]
            val childNode = node.children[ch] ?: continue
            dfs(x, y, childNode, output + ch, visited)
        }
        visited[row][col] = false
    }
    println(board.joinToString("\n") { String(it) })
    for (i in 0 until m) {
        for (j in 0 until n) {
            val ch = board[i][j]
            val node = trie.root.children[ch] ?: continue
            dfs(i, j, node, ch.toString(), Array(m) { BooleanArray(n) })
        }
    }
    return result.toList()
}

fun exist(board: Array<CharArray>, word: String): Boolean {
    val trie = Trie()
    trie.insert(word)
    val maxLength = word.length
    val m = board.size
    val n = board[0].size
    var found = false
    //  val visited = Array(m) { BooleanArray(n) }
    val dirX = intArrayOf(1, -1, 0, 0)
    val dirY = intArrayOf(0, 0, 1, -1)
    fun dfs(row: Int, col: Int, node: Trie.TrieNode, output: String, visited: Array<BooleanArray>) {
        visited[row][col] = true
        if (found) return
        if (node.isWord) {
            found = true
            return
        }
        if (output.length >= maxLength) {
            visited[row][col] = false
            return
        }
        for (i in 0 until 4) {
            val x = row + dirX[i]
            val y = col + dirY[i]
            if (x !in 0 until m || y !in 0 until n) continue
            if (visited[x][y]) continue
            val ch = board[x][y]
            val childNode = node.children[ch] ?: continue
            dfs(x, y, childNode, output + ch, visited)
        }
        visited[row][col] = false
    }
    //  println(board.joinToString("\n") { String(it) })
    for (i in 0 until m) {
        for (j in 0 until n) {
            val ch = board[i][j]
            val node = trie.root.children[ch] ?: continue
            dfs(i, j, node, ch.toString(), Array(m) { BooleanArray(n) })
        }
    }
    return found
}

fun sumPrefixScores(words: Array<String>): IntArray {
    class Node {
        val children = mutableMapOf<Char, Node>()
        var isWord = false
        var count = 0
    }

    val wordList = words.withIndex()
        .sortedWith(comparator = compareBy<IndexedValue<String>> { it.value.length }.thenBy { it.value })
  //  println(wordList.toList())
    val root = Node()
    for (entry in wordList) {
        val word = entry.value
        var node = root
        for (c in word) {
            node = node.children.computeIfAbsent(c) { Node() }
            node.count++
        }
        node.isWord = true
    }

    val answers = IntArray(words.size)
    for (i in words.indices) {
        val word = words[i]
        var cnt = 0
      //  println("word: $word")
        var node = root
        var minSoFar = Int.MAX_VALUE
        for (c in word) {
            node = node.children[c] ?: break
            minSoFar = minOf(minSoFar, node.count)
         //   print("${node.count} ")
            cnt += minSoFar
        }
      //  println()
        answers[i] = cnt
    }

    return answers

}

fun main() {
    println(
        sumPrefixScores(arrayOf("abc", "ab", "bc", "b")).toList()
    )
}