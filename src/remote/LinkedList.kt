package remote

fun sortList(head: ListNode?): ListNode? {
    if (head?.next == null) return head

    var fastPointer = head.next
    var slowPointer = head
    while (fastPointer != null && fastPointer.next != null) {
        fastPointer = fastPointer.next?.next
        slowPointer = slowPointer?.next
    }
    val middle = slowPointer
    val nextToMiddle = middle?.next
    middle?.next = null

    var left = sortList(head)
    var right = sortList(nextToMiddle)

    val root = ListNode(0)
    var tail: ListNode? = root

    while (left != null && right != null) {
        if (left.`val` < right.`val`) {
            tail?.next = left
            left = left.next
        } else {
            tail?.next = right
            right = right.next
        }
        tail = tail?.next
    }
    if (left != null) tail?.next = left
    if (right != null) tail?.next = right
    return root.next
}

fun sortArray(nums: IntArray): IntArray {
    return quickSortArray(nums, 0, nums.size - 1)
}

fun quickSortArray(nums: IntArray, start: Int, end: Int): IntArray {
    val mid = (start + end) / 2
    val pivot = nums[mid]
    var l = start
    var r = end
    while (l <= r) {
        while (nums[l] < pivot) l++
        while (nums[r] > pivot) r--
        if (l <= r) {
            val tmp = nums[l]
            nums[l] = nums[r]
            nums[r] = tmp
            l++
            r--
        }
    }
    if (l < end) quickSortArray(nums, l, end)
    if (start < r) quickSortArray(nums, start, r)
    return nums
}

fun modifiedList(nums: IntArray, head: ListNode?): ListNode? {
    val set = nums.toSet()
    val root = ListNode(0)
    var tail: ListNode? = root
    var pointer: ListNode? = head

    while (pointer != null) {
        val value = pointer.`val`
        if (value !in set) {
            tail?.next = pointer
            tail = tail?.next
        }
        pointer = pointer.next
    }
    tail?.next = null
    return root.next
}

fun deleteNode(node: ListNode?) {
    node ?: return
    node.`val` = node.next?.`val` ?: 0
    node.next = node.next?.next
}

fun removeNthFromEnd(head: ListNode?, n: Int): ListNode? {
    val root = ListNode(0)
    root.next = head
    var first: ListNode? = root
    var second: ListNode? = root
    for (i in 0..n) {
        second = second?.next
    }

    while (second != null) {
        second = second.next
        first = first?.next
    }
    first?.next = first.next?.next
    return root.next
}

fun deleteMiddle(head: ListNode?): ListNode? {
    val root = ListNode(0)
    root.next = head

    var slow: ListNode? = root
    var fast = root.next

    while (fast != null && fast.next != null) {
        slow = slow?.next
        fast = fast.next?.next
    }
    slow?.next = slow.next?.next
    return root.next
}

fun rotateRight(head: ListNode?, k: Int): ListNode? {
    if (head == null) return null
    var size = 1
    var lastNode = head

    while (lastNode?.next != null) {
        lastNode = lastNode.next
        size++
    }

    val p = k % size
    if (p == 0) return head
    var cut = size - p

    var newLast = head
    while (--cut > 0) {
        newLast = newLast?.next
    }

    val newHead = newLast?.next
    newLast?.next = null
    lastNode?.next = head
    return newHead
}

fun splitListToParts(head: ListNode?, k: Int): Array<ListNode?> {
    if (head == null) return arrayOfNulls(k)
    var size = 1
    var lastNode = head
    while (lastNode?.next != null) {
        lastNode = lastNode.next
        size++
    }

    val bucketSize = size / k
    val extraSize = size % k

    println("bucket $bucketSize, $extraSize")

    var node = head
    return Array(k) { id ->
        var targetSize = bucketSize + if (id + 1 <= extraSize) 1 else 0
        if (targetSize == 0) return@Array null
        val partHead = node
        while (--targetSize > 0) {
            node = node?.next
        }
        val partLast = node
        node = node?.next
        partLast?.next = null
        partHead
    }
}

fun addTwoNumbers(l1: ListNode?, l2: ListNode?): ListNode? {
//    var first: ListNode? = null
//    var second: ListNode? = null
//    var a = l1
//    while (a != null){
//        val node = ListNode(a.`val`)
//        node.next = first
//        first = node
//        a = a.next
//    }
//
//    var b = l2
//    while (b != null){
//        val node = ListNode(b.`val`)
//        node.next = second
//        second = node
//        b = b.next
//    }
//
//    var result: ListNode? = null
//    var carry = 0
//
//    while (first != null || second != null || carry > 0) {
//        val x = first?.`val` ?: 0
//        val y = second?.`val` ?: 0
//        val value = x + y + carry
//        carry = value / 10
//        val node = ListNode(value % 10)
//        node.next = result
//        first = first?.next
//        second = second?.next
//        result = node
//    }
//    return result
    var first: ListNode? = l1
    var second: ListNode? = l2
    val root = ListNode(0)
    var result: ListNode? = root
    var carry = 0

    while (first != null || second != null || carry > 0) {
        val x = first?.`val` ?: 0
        val y = second?.`val` ?: 0
        val value = x + y + carry
        carry = value / 10
        val node = ListNode(value % 10)
        result?.next = node
        result = result?.next
        first = first?.next
        second = second?.next
    }
    return root.next
}

fun main() {
    println(
        createSortedArray(intArrayOf(1, 5, 6, 2))
    )
}