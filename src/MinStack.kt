import java.util.Stack
import kotlin.math.min

class MinStack() {
    val stack = Stack<Int>()
    val mins = ArrayDeque<Int>()

    fun push(`val`: Int) {
        stack.add(`val`)
        if (mins.isEmpty()) {
            mins.addLast(`val`)
        } else {
            mins.addLast(minOf(mins.last(), `val`))
        }

    }

    fun pop() {
        if (stack.isNotEmpty()) {
            stack.pop()
        }
        mins.removeLastOrNull()

    }

    fun top(): Int {
        return stack.peek()
    }

    fun getMin(): Int {
        return mins.lastOrNull() ?: Int.MIN_VALUE
    }

}