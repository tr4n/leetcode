package local

import java.util.*

class FoodRatings(
    private val foods: Array<String>,
    private val cuisines: Array<String>,
    private val ratings: IntArray
) {

    private val n = foods.size
    private val foodToIndex = mutableMapOf<String, Int>()
    private val ratingTree = mutableMapOf<String, TreeSet<String>>()

    init {
        group()
    }

    private fun group() {
        for (i in 0 until n) {
            val cuisine = cuisines[i]
            val food = foods[i]
            // val rating = ratings[i]
            foodToIndex[food] = i
            ratingTree.computeIfAbsent(cuisine) {
                TreeSet(compareBy<String> { ratings[foodToIndex[it] ?: 0] }.thenByDescending { it })
            }.add(food)
        }
    }

    fun changeRating(food: String, newRating: Int) {
        val index = foodToIndex[food] ?: return
        val cuisine = cuisines[index]
        ratingTree[cuisine]?.remove(food)
        ratings[index] = newRating
        ratingTree[cuisine]?.add(food)

       //   println(ratingTree[cuisine])
    }

    fun highestRated(cuisine: String): String {
        val result = ratingTree[cuisine]?.last() ?: ""
       //    println(result)
        return result
    }
}

fun main() {
    val foodRatings = FoodRatings(
        arrayOf("kimchi", "miso", "sushi", "moussaka", "ramen", "bulgogi"),
        arrayOf("korean", "japanese", "japanese", "greek", "japanese", "korean"),
        intArrayOf(9, 12, 8, 15, 14, 7),
    )
    foodRatings.highestRated("korean"); // return "kimchi"
    // "kimchi" is the highest rated korean food with a rating of 9.
    foodRatings.highestRated("japanese"); // return "ramen"
    // "ramen" is the highest rated japanese food with a rating of 14.
    foodRatings.changeRating("sushi", 16); // "sushi" now has a rating of 16.
    foodRatings.highestRated("japanese"); // return "sushi"
    // "sushi" is the highest rated japanese food with a rating of 16.
    foodRatings.changeRating("ramen", 16); // "ramen" now has a rating of 16.
    foodRatings.highestRated("japanese");
}