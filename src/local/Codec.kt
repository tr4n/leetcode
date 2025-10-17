package local

import topic.TreeNode
import java.util.*

class Codec() {
    private val base = "http://tinyurl.com/"
    private val urlToId = mutableMapOf<String, String>()
    private val idToUrl = mutableMapOf<String, String>()

    fun encode(longUrl: String): String {
        var id = urlToId[longUrl]
        if (id != null) return base + id
        id = urlToId.size.toString()
        urlToId[longUrl] = id
        idToUrl[id] = longUrl
        return base + id
    }

    // Decodes a shortened URL to its original URL.
    fun decode(shortUrl: String): String {
        val id = shortUrl.substringAfter(base)
        return idToUrl[id] ?: ""
    }
}

fun deepestLeavesSum(root: TreeNode?): Int {
    val depth = mutableListOf<Int>()

    fun dfs(node: TreeNode?, level: Int) {
        if (node == null) return
        if (level == depth.size) {
            depth.add(0)
        }
        depth[level] += node.`val`
        dfs(node.left, level + 1)
        dfs(node.right, level + 1)
    }
    dfs(root, 0)
    return depth.lastOrNull() ?: 0
}

fun findMatrix(nums: IntArray): List<List<Int>> {
    val grid = mutableListOf(mutableSetOf<Int>())

    for (num in nums) {
        var i = 0
        while (i < grid.size) {
            if (num !in grid[i]) break
            i++
        }
        if (i == grid.size) {
            grid.add(mutableSetOf(num))
        } else {
            grid[i].add(num)
        }
    }
    return grid.map { it.toList() }
}

fun maxIncreaseKeepingSkyline(grid: Array<IntArray>): Int {
    val m = grid.size
    val n = grid[0].size
    val maxRows = IntArray(m)
    val maxCols = IntArray(n)

    for (i in 0 until m) {
        for (j in 0 until n) {
            maxRows[i] = maxOf(maxRows[i], grid[i][j])
            maxCols[j] = maxOf(maxCols[j], grid[i][j])
        }
    }
    var sum = 0
    for (i in 0 until m) {
        for (j in 0 until n) {
            sum += minOf(maxRows[i], maxCols[j]) - grid[i][j]
        }
    }
    return sum
}

fun findFarmland(land: Array<IntArray>): Array<IntArray> {
    val m = land.size
    val n = land[0].size

    fun fill(row: Int, col: Int, id: Int) {
        if (row !in 0 until m || col !in 0 until n) return
        if (land[row][col] != 1) return
        land[row][col] = id
        fill(row + 1, col, id)
        fill(row, col + 1, id)
        fill(row - 1, col, id)
        fill(row, col - 1, id)
    }

    var groups = 0
    for (i in 0 until m) {
        for (j in 0 until n) {
            if (land[i][j] == 1) {
                fill(i, j, --groups)
            }
        }
    }
    if (groups == 0) return emptyArray()

    val result = Array(groups) { intArrayOf(m, n, -1, -1) }
    for (i in 0 until m) {
        for (j in 0 until n) {
            if (land[i][j] == 0) continue
            val id = -1 - land[i][j]
            result[id][0] = minOf(result[id][0], i)
            result[id][1] = minOf(result[id][1], j)
            result[id][2] = maxOf(result[id][2], i)
            result[id][3] = maxOf(result[id][3], j)

        }
    }
    return result
}

fun goodNodes(root: TreeNode?): Int {

    var cnt = 0
    fun dfs(node: TreeNode?, maxSoFar: Int) {
        if (node == null) return
        if (node.`val` >= maxSoFar) {
            cnt++
        }
        val newMax = maxOf(maxSoFar, node.`val`)
        dfs(node.left, newMax)
        dfs(node.right, newMax)
    }
    dfs(root, Int.MIN_VALUE)
    return cnt
}

fun diffWaysToCompute(expression: String): List<Int> {
    val n = expression.length
    val result = mutableListOf<Int>()
    for (i in 0 until n) {
        val ch = expression[i]
        if (ch !in "+-*") continue
        val leftExpr = expression.substring(0, i)
        val rightExpr = expression.substring(i + 1)
        val left = diffWaysToCompute(leftExpr)
        val right = diffWaysToCompute(rightExpr)

        for (l in left) {
            for (r in right) {
                val value = when (ch) {
                    '+' -> l + r
                    '-' -> l - r
                    '*' -> l * r
                    else -> 0
                }
                result.add(value)
            }
        }
    }
    if (result.isEmpty()) {
        result.add(expression.toInt())
    }
    return result
}

fun addOperators(num: String, target: Int): List<String> {
    val operators = setOf('+', '-', '*')
    fun precedence(op: Char): Long = when (op) {
        '+', '-' -> 1
        '*' -> 2
        else -> 0
    }

    fun applyOp(a: Long, b: Long, op: Char): Long {
        return when (op) {
            '+' -> a + b
            '-' -> a - b
            '*' -> a * b
            else -> 0
        }
    }

    fun evalExpr(expr: String): Long {
        val values = ArrayDeque<Long>()
        val ops = ArrayDeque<Char>()

        var i = 0
        while (i < expr.length) {
            val ch = expr[i]

            when {
                ch == '(' -> ops.addLast(ch)

                ch.isDigit() -> {
                    var num = 0L
                    while (i < expr.length && expr[i].isDigit()) {
                        num = num * 10L + (expr[i] - '0').toLong()
                        i++
                    }
                    i--
                    values.addLast(num)
                }

                ch == ')' -> {
                    while (ops.isNotEmpty() && ops.last() != '(') {
                        if (values.size < 2) return 0
                        val b = values.removeLast()
                        val a = values.removeLast()
                        val op = ops.removeLast()
                        values.addLast(applyOp(a, b, op))
                    }
                    if (ops.isNotEmpty() && ops.last() == '(') ops.removeLast()
                }

                ch in operators -> {
                    while (ops.isNotEmpty() && precedence(ops.last()) >= precedence(ch)) {
                        if (values.size < 2) return 0
                        val b = values.removeLast()
                        val a = values.removeLast()
                        val op = ops.removeLast()
                        values.addLast(applyOp(a, b, op))
                    }
                    ops.addLast(ch)
                }
            }
            i++
        }

        while (ops.isNotEmpty()) {
            if (values.size < 2) return 0
            val b = values.removeLast()
            val a = values.removeLast()
            val op = ops.removeLast()
            values.addLast(applyOp(a, b, op))
        }

        return if (values.size == 1) values.last() else 0
    }

    val n = num.length
    val result = mutableListOf<String>()
    val expr = StringBuilder()
    expr.append(num[0])
    fun dfs(pos: Int) {
        if (pos == n) {
            val expression = expr.toString()
            val value = evalExpr(expression)
            if (value == target.toLong()) {
                result.add(expression)
            }
            return
        }
        val digit = num[pos]
        for (op in operators) {
            expr.append(op)
            expr.append(digit)
            dfs(pos + 1)
            expr.deleteCharAt(expr.length - 1)
            expr.deleteCharAt(expr.length - 1)
        }

        var i = expr.length - 1
        var head = expr[i]
        while (i >= 0 && expr[i] !in operators) {
            head = expr[i--]
        }

        if (head != '0') {
            expr.append(digit)
            dfs(pos + 1)
            expr.deleteCharAt(expr.length - 1)
        }

    }

    dfs(1)
    return result
}


fun calculate(s: String): Int {
    val operators = setOf('+', '-', '*')
    fun precedence(op: Char): Long = when (op) {
        '+', '-' -> 1
        '*' -> 2
        else -> 0
    }

    fun applyOp(a: Long, b: Long, op: Char): Long {
        return when (op) {
            '+' -> a + b
            '-' -> a - b
            '*' -> a * b
            else -> 0
        }
    }

    fun evalExpr(expr: String): Long {
        val values = ArrayDeque<Long>()
        val ops = ArrayDeque<Char>()

        var i = 0
        while (i < expr.length) {
            val ch = expr[i]

            when {
                ch == '(' -> ops.addLast(ch)

                ch.isDigit() -> {
                    var num = 0L
                    while (i < expr.length && expr[i].isDigit()) {
                        num = num * 10L + (expr[i] - '0').toLong()
                        i++
                    }
                    i--
                    values.addLast(num)
                }

                ch == ')' -> {
                    while (ops.isNotEmpty() && ops.last() != '(') {
                        if (values.size < 2) return 0
                        val b = values.removeLast()
                        val a = values.removeLast()
                        val op = ops.removeLast()
                        values.addLast(applyOp(a, b, op))
                    }
                    if (ops.isNotEmpty() && ops.last() == '(') ops.removeLast()
                }

                ch in operators -> {
                    while (ops.isNotEmpty() && precedence(ops.last()) >= precedence(ch)) {
                        if (values.size < 2) return 0
                        val b = values.removeLast()
                        val a = values.removeLast()
                        val op = ops.removeLast()
                        values.addLast(applyOp(a, b, op))
                    }
                    ops.addLast(ch)
                }
            }
            i++
        }

        while (ops.isNotEmpty()) {
            if (values.size < 2) return 0
            val b = values.removeLast()
            val a = values.removeLast()
            val op = ops.removeLast()
            values.addLast(applyOp(a, b, op))
        }

        return if (values.size == 1) values.last() else 0
    }

    val str = StringBuilder()
    for (c in s) if (c != ' ') str.append(c)
    val expr = StringBuilder()
    var i = 0
    while (i < str.length) {
        val ch = s[i]
        if (ch == '-') {
            if (i == 0 || s[i - 1] == '(' || s[i - 1] == '+' || s[i - 1] == '-') {
                expr.append('0')
            }
        }
        if (ch != ' ') {
            expr.append(ch)
        }
        i++
    }
    return evalExpr(expr.toString()).toInt()
}

fun minimizeResult(expression: String): String {
    val (left, right) = expression.split("+")
    val leftNum = left.toInt()
    val rightNum = right.toInt()
    var minValue = leftNum + rightNum
    var minExpr = "($left+$right)"
    for (i in 1 until left.length) {
        val a = left.substring(0, i).toInt()
        val b = left.substring(i).toInt()

        for (j in 1 until right.length) {
            val c = right.substring(0, j).toInt()
            val d = right.substring(j).toInt()
            val value = a * (b + c) * d
            if (value < minValue) {
                minValue = value
                minExpr = "$a($b+$c)$d"
            }
        }

        val value = a * (b + rightNum)
        if (value < minValue) {
            minValue = value
            minExpr = "$a($b+$right)"
        }
    }

    for (j in 1 until right.length) {
        val c = right.substring(0, j).toInt()
        val d = right.substring(j).toInt()
        val value = (leftNum + c) * d
        if (value < minValue) {
            minValue = value
            minExpr = "($left+$c)*$d)"
        }
    }
    return minExpr
}

fun solveEquation2(equation: String): String {
    val operators = setOf('+', '-', '*')
    fun precedence(op: Char): Long = when (op) {
        '+', '-' -> 1
        '*' -> 2
        else -> 0
    }

    fun applyOp(a: Long, b: Long, op: Char): Long {
        return when (op) {
            '+' -> a + b
            '-' -> a - b
            '*' -> a * b
            else -> 0
        }
    }

    fun evalExpr(expr: String): Long {
        val values = ArrayDeque<Long>()
        val ops = ArrayDeque<Char>()

        var i = 0
        while (i < expr.length) {
            val ch = expr[i]

            when {
                ch == '(' -> ops.addLast(ch)

                ch.isDigit() -> {
                    var num = 0L
                    while (i < expr.length && expr[i].isDigit()) {
                        num = num * 10L + (expr[i] - '0').toLong()
                        i++
                    }
                    i--
                    values.addLast(num)
                }

                ch == ')' -> {
                    while (ops.isNotEmpty() && ops.last() != '(') {
                        if (values.size < 2) return 0
                        val b = values.removeLast()
                        val a = values.removeLast()
                        val op = ops.removeLast()
                        values.addLast(applyOp(a, b, op))
                    }
                    if (ops.isNotEmpty() && ops.last() == '(') ops.removeLast()
                }

                ch in operators -> {
                    while (ops.isNotEmpty() && precedence(ops.last()) >= precedence(ch)) {
                        if (values.size < 2) return 0
                        val b = values.removeLast()
                        val a = values.removeLast()
                        val op = ops.removeLast()
                        values.addLast(applyOp(a, b, op))
                    }
                    ops.addLast(ch)
                }
            }
            i++
        }

        while (ops.isNotEmpty()) {
            if (values.size < 2) return 0
            val b = values.removeLast()
            val a = values.removeLast()
            val op = ops.removeLast()
            values.addLast(applyOp(a, b, op))
        }

        return if (values.size == 1) values.last() else 0
    }

    val (left, right) = equation.split("=")
    val function = "$left-($right)"
    val inf = 1_000_000L
    var lo = -inf
    var hi = inf

    fun f(x: Long): Long = evalExpr(function.replace("x", "$x"))

    var fLo = f(lo)
    var fHi = f(hi)

    while (fLo * fHi > 0) {
        lo *= 2
        hi *= 2
        fLo = f(lo)
        fHi = f(hi)
        if (lo < -1e12 || hi > 1e12) return "Infinite solutions"
    }

    var ans: Long? = null
    while (lo <= hi) {
        val mid = (lo + hi) / 2
        val fMid = f(mid)
        if (fMid == 0L) {
            ans = mid
            break
        }
        if (fLo * fMid <= 0) {
            hi = mid - 1
            fHi = fMid
        } else {
            lo = mid + 1
            fLo = fMid
        }
    }

    return ans?.toString() ?: "Infinite solutions"
}

fun solveEquation(equation: String): String {
    fun shortExpression(expr: String): Pair<Int, Int> {
        val terms = expr.split(Regex("(?=[+-])")).filter { it.isNotBlank() }

        var coef = 0
        var constant = 0

        for (term in terms) {
            if (term.contains("x")) {
                val t = term.replace("x", "")
                coef += when (t) {
                    "", "+" -> 1
                    "-" -> -1
                    else -> t.toInt()
                }
            } else {
                constant += term.toInt()
            }
        }
        return coef to constant
    }
    val (left, right) = equation.split("=")
    val (leftCoef, leftConst) = shortExpression(left)
    val (rightCoef, rightConst) = shortExpression(right)
    val a = leftCoef - rightCoef
    val b = rightConst - leftConst
    return when {
        a == 0 && b == 0 -> "Infinite solutions"
        a == 0 && b != 0 -> "No solution"
        else -> "${b / a}"
    }
}
