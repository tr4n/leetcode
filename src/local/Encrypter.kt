package local

import kotlin.text.iterator

class SimpleTrie {
    val root = Node(0)
    private var nodeCount = 0

    class Node(val id: Int) {
        val children = mutableMapOf<Char, Node>()
        var isWord: Boolean = false
    }

    fun insert(word: String) {
        var node = root
        for (c in word) {
            node = node.children.computeIfAbsent(c) { Node(++nodeCount) }
        }
        node.isWord = true
    }
}

class Encrypter(
    private val keys: CharArray,
    private val values: Array<String>,
    dictionary: Array<String>
) {

    private val keyToValue =
        keys.withIndex().associate { it.value to (values.getOrNull(it.index)) }
    private val valueToKey = mutableMapOf<String, MutableList<Char>>()
    private val trie = SimpleTrie()

    init {
        buildValueToKey()
        for (word in dictionary) {
            trie.insert(word)
        }
    }

    private fun buildValueToKey() {
        for (i in 0 until values.size) {
            val value = values[i]
            val key = keys[i]
            if (valueToKey[value] == null) {
                valueToKey[value] = mutableListOf(key)
            } else {
                valueToKey[value]?.add(key)
            }
        }
    }

    fun encrypt(word1: String): String {
        val result = StringBuilder()
        for (c in word1) {
            val ch = keyToValue[c] ?: return ""
            result.append(ch)
        }
        return result.toString()
    }

    fun decrypt(word2: String): Int {
        return decode(word2, 0, trie.root, mutableMapOf())
    }

    private fun decode(
        word2: String,
        pos: Int,
        node: SimpleTrie.Node,
        memo: MutableMap<Pair<Int, SimpleTrie.Node>, Int>
    ): Int {
        val state = pos to node
        memo[state]?.let { return it }

        if (pos > word2.length) return 0
        if (pos == word2.length) {
            val res = if (node.isWord) 1 else 0
            memo[state] = res
            return res
        }

        val sub = word2.substring(pos, pos + 2)
        val keys = valueToKey[sub] ?: return 0
        var result = 0
        for (key in keys) {
            val childNode = node.children[key] ?: continue
            result += decode(word2, pos + 2, childNode, memo)
        }

        memo[state] = result
        return result
    }
}

fun main() {
    val encrypter = Encrypter(
        charArrayOf('a', 'b', 'c', 'z'),
        arrayOf("aa", "bb", "cc", "zz"),
        arrayOf("aa", "aaa", "aaaa", "aaaaa", "aaaaaaa")
    )
    println(
        encrypter.decrypt("aa")
    )
}