package local

import java.util.Stack

class BrowserHistory(homepage: String) {
    private val previous = Stack<String>()
    private val next = Stack<String>()

    init {
        previous.add(homepage)
    }


    fun visit(url: String) {
        previous.add(url)
        next.clear()

    }

    fun back(steps: Int): String {
        var cnt = steps
        while (cnt > 0 && previous.size > 1) {
            next.add(previous.pop())
            cnt--
        }
        return previous.peek()
    }

    fun forward(steps: Int): String {
        var cnt = steps
        while (cnt > 0 && next.isNotEmpty()) {
            previous.add(next.pop())
            cnt--
        }
        return previous.peek()
    }

}