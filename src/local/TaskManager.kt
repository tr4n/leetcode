package local

import java.util.*

class TaskManager(tasks: List<List<Int>>) {
    class Task(val userId: Int, val taskId: Int, var priority: Int)

    private val taskMap = mutableMapOf<Int, Task>()
    private val treeSet = TreeSet<Task>(
        compareBy<Task> { it.priority }
            .thenBy { it.taskId }
            .thenBy { it.userId }
    )

    init {
        for ((userId, taskId, priority) in tasks) {
            val task = Task(userId, taskId, priority)
            taskMap[taskId] = task
            treeSet.add(task)
        }
    }

    fun add(userId: Int, taskId: Int, priority: Int) {
        val task = Task(userId, taskId, priority)
        taskMap[taskId] = task
        treeSet.add(task)
    }

    fun edit(taskId: Int, newPriority: Int) {
        val task = taskMap[taskId] ?: return
        treeSet.remove(task)
        task.priority = newPriority
        taskMap[taskId] = task
        treeSet.add(task)
    }

    fun rmv(taskId: Int) {
        val task = taskMap[taskId] ?: return
        treeSet.remove(task)
        taskMap.remove(taskId)
    }

    fun execTop(): Int {
        if (treeSet.isEmpty()) return -1
        val task = treeSet.pollLast() ?: return -1
        treeSet.remove(task)
        taskMap.remove(task.taskId)
        return task.userId
    }

}

fun main(){
    val taskManager = TaskManager("[[[1,101,10],[2,102,20],[3,103,15]]".to2DIntArray().map { it.toList() })
    taskManager.add(4, 104, 5); // Adds task 104 with priority 5 for User 4.
    taskManager.edit(102, 8); // Updates priority of task 102 to 8.
    taskManager.execTop(); // return 3. Executes task 103 for User 3.
    taskManager.rmv(101); // Removes task 101 from the system.
    taskManager.add(5, 105, 15); // Adds task 105 with priority 15 for User 5.
    taskManager.execTop(); // return 5. Executes task 105 for User 5.
}