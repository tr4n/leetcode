package local


fun partitionString(s: String): List<String> {
    class Node {
        val children = mutableMapOf<Char, Node>()
    }

    val root = Node()

    val set = mutableListOf<String>()
    var node: Node = root
    var word = StringBuilder()
    for (c in s) {
        val nextNode = node.children[c]
        word.append(c)
        if (nextNode == null) {
            node.children[c] = Node()
            node = root
            set.add(word.toString())
            word = StringBuilder()
        } else {
            node = nextNode
        }

    }

    return set
}

fun minExtraChar(s: String, dictionary: Array<String>): Int {
    class Node {
        val children = mutableMapOf<Char, Node>()
        var isWord = false
    }

    val root = Node()

    for (word in dictionary) {
        var node = root
        for (c in word) {
            node = node.children.computeIfAbsent(c) { Node() }
        }
        node.isWord = true
    }

    val covered = mutableSetOf<Int>()
    for (i in 0 until s.length) {
        var j = i
        var node = root
        var previous = i
        while (j < s.length) {
            val ch = s[j++]
            node = node.children[ch] ?: break
            if (node.isWord) {
                covered.addAll(previous until j)
                previous = j
            }
        }
    }
    return s.length - covered.size
}

fun wordBreak(s: String, wordDict: List<String>): List<String> {
    val n = s.length

    class Node {
        val children = mutableMapOf<Char, Node>()
        var isWord = false
    }

    val root = Node()

    for (word in wordDict) {
        var node = root
        for (c in word) {
            node = node.children.computeIfAbsent(c) { Node() }
        }
        node.isWord = true
    }

    val result = mutableListOf<String>()
    val path = mutableListOf<String>()
    val memo = mutableMapOf<Int, List<String>>()

    fun backtrack(start: Int) {
        if (start == n) {
            result.add(path.joinToString(" "))
            return
        }
        var node = root
        val builder = StringBuilder()
        for (i in start until n) {
            val ch = s[i]
            builder.append(ch)
            node = node.children[ch] ?: break
            if (node.isWord) {
                path.add(builder.toString())
                backtrack(i + 1)
                path.removeLast()
            }
        }
    }

    backtrack(0)
    return result
}

fun findAllConcatenatedWordsInADict(words: Array<String>): List<String> {
    class Node {
        val children = mutableMapOf<Char, Node>()
        var isWord = false
    }

    val result = mutableListOf<String>()
    words.sortWith(comparator = compareBy<String> { it.length }.thenBy { it })
    val root = Node()

    for (word in words) {
        var node = root
        for (c in word) {
            node = node.children.computeIfAbsent(c) { Node() }
        }
        node.isWord = true
    }


    fun dfs(word: String, pos: Int, cnt: Int, memo: MutableMap<Int, Boolean>): Boolean {
        //   if (word in result) return true
        if (pos == word.length) return cnt >= 2
        val cache = memo[pos]
        if (cache != null) return cache

        var node = root
        for (i in pos until word.length) {
            val c = word[i]
            node = node.children[c] ?: return false
            if (node.isWord) {
                val next = dfs(word, i + 1, cnt + 1, memo)
                if (next) {
                    memo[pos] = true
                    return true
                }
            }
        }
        memo[pos] = false
        return false
    }

    fun canBeFormed(word: String, dict: Set<String>): Boolean {
        val dp = BooleanArray(word.length + 1)
        dp[0] = true
        for (i in 1..word.length) {
            for (j in 0 until i) {
                if (!dp[j]) continue
                val sub = word.substring(j, i)
                if (sub in dict) {
                    dp[i] = true
                    break
                }
            }
        }
        return dp[word.length]
    }

    val dict = mutableSetOf<String>()
    for (word in words) {
        if(canBeFormed(word, dict)) {
            result.add(word)
        }
        dict.add(word)
    }
    return result.toList()
}

fun main() {
    println(
        findAllConcatenatedWordsInADict(
            arrayOf(
                "cat", "cats", "catsdogcats", "dog", "dogcatsdog", "hippopotamuses", "rat", "ratcatdogcat"
            )
        )
    )
}