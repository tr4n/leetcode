package local

import kotlin.text.iterator

class Encrypter(keys: CharArray, values: Array<String>, private val dictionary: Array<String>) {

    private val keyToValue =
        keys.withIndex().associate { it.value to (values.getOrNull(it.index) ) }
    private val valueToKey =
        values.withIndex().associate { it.value to (keys.getOrNull(it.index) ) }
    private val dictPrefixes = buildPrefixes()

    private fun buildPrefixes(): Set<String> {
        val set =  mutableSetOf<String>()
        for(s in dictionary) {
            val builder = StringBuilder()
            for(c in s) {
                builder.append(c)
                set.add(builder.toString())
            }
        }
        return set
    }

    fun encrypt(word1: String): String {
        val result = StringBuilder()
        for(c in word1) {
            val ch = keyToValue[c] ?: return ""
            result.append(ch)
        }
        return result.toString()
    }

    fun decrypt(word2: String): Int {
        val result = StringBuilder()
        for(c in word1) {
            val ch = keyToValue[c] ?: return ""
            result.append(ch)
        }
        return result.toString()
    }

}