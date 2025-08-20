package org.example

class ProductOfNumbers() {
    var prefixProduct = mutableListOf<Int>()
    var result = 1

    val queue = ArrayDeque<Int>()

    fun add(num: Int) {
        if(num == 0) {
            prefixProduct = mutableListOf()
            result = 1
        } else {
            result *= num
            prefixProduct.add(result)
        }
    }

    fun getProduct(k: Int): Int {
        val size = prefixProduct.size
        if(size < k) return 0
        val previous = if (size < k + 1) 1 else prefixProduct[size - k - 1]
        return prefixProduct.last() / previous
    }
}
