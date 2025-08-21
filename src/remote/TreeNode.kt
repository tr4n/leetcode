package remote

class TreeNode(var `val`: Int) {
    var left: TreeNode? = null
    var right: TreeNode? = null
}

fun buildGraphIterative(root: TreeNode?): Pair<List<Int>, List<MutableList<Int>>> {
    val nodes = mutableListOf<Int>()
    val edges = mutableListOf<MutableList<Int>>()
    val nodeToIndex = mutableMapOf<TreeNode, Int>()

    if (root == null) return Pair(nodes, edges)

    val stack = ArrayDeque<TreeNode>()
    stack.add(root)

    while (stack.isNotEmpty()) {
        val node = stack.removeLast()

        val index = nodeToIndex.getOrPut(node) {
            val newIndex = nodes.size
            nodes.add(node.`val`)
            edges.add(mutableListOf())
            newIndex
        }

        val childNodes = listOfNotNull(node.right, node.left)

        for(childNode in childNodes) {
            val nodeIndex = nodeToIndex.getOrPut(childNode) {
                val newIndex = nodes.size
                nodes.add(childNode.`val`)
                edges.add(mutableListOf())
                newIndex
            }
            edges[index].add(nodeIndex)
            stack.add(childNode)
        }
    }

    return Pair(nodes, edges)
}

fun buildTree(values: List<Int?>): TreeNode? {
    if (values.isEmpty() || values[0] == null) return null

    val root = TreeNode(values[0]!!)
    val queue = ArrayDeque<TreeNode>()
    queue.add(root)

    var i = 1
    while (i < values.size) {
        val current = queue.removeFirst()

        // Gán left
        if (i < values.size && values[i] != null) {
            val leftNode = TreeNode(values[i]!!)
            current.left = leftNode
            queue.add(leftNode)
        }
        i++

        // Gán right
        if (i < values.size && values[i] != null) {
            val rightNode = TreeNode(values[i]!!)
            current.right = rightNode
            queue.add(rightNode)
        }
        i++
    }

    return root
}