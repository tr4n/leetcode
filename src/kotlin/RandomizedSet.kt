package org.example

class RandomizedSet() {
    private val set = mutableSetOf<Int>()

    fun insert(`val`: Int): Boolean {
        return set.add(`val`)
    }

    fun remove(`val`: Int): Boolean {
        return set.remove(`val`)
    }

    fun getRandom(): Int {
        return set.random()
    }
}

