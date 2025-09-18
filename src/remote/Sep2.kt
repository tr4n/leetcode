package remote

fun displayTable(orders: List<List<String>>): List<List<String>> {
    val tables = mutableSetOf<String>()
    val foods = mutableSetOf<String>()
    val map = mutableMapOf<String, MutableMap<String, Int>>()
  //  val sum = mutableMapOf<String, Int>()

    for ((_, table, food) in orders) {
        tables.add(table)
        foods.add(food)
        val quantity = map.computeIfAbsent(table) { mutableMapOf() }[food] ?: 0
        map[table]?.set(food, quantity + 1)
//        sum[table] = (sum[table] ?: 0) + 1
    }


    val foodList = foods.sorted()
    val tableList = tables.sortedBy { it.toInt() }

    val result = mutableListOf<MutableList<String>>()
    result.add(mutableListOf())
    result[0].add("Table")
    result[0].addAll(foodList)

    for(table in tableList) {
        val list = mutableListOf<String>()
        list.add(table)
        for(food in foodList) {
            list.add((map[table]?.get(food) ?: 0).toString())
        }
        result.add(list)
    }

    return result
}