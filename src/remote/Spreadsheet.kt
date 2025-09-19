package remote

class Spreadsheet(rows: Int) {
    val map = mutableMapOf<String, Int>()

    fun setCell(cell: String, value: Int) {
        map[cell] = value
    }

    fun resetCell(cell: String) {
        map[cell] = 0
    }

    fun getValue(formula: String): Int {
        val (first, second) = formula.removePrefix("=").split("+")
        val firstValue = first.toIntOrNull() ?: map[first] ?: 0
        val secondValue = second.toIntOrNull() ?: map[second] ?: 0
        // println("$first = $firstValue, $second = $secondValue")
        return firstValue + secondValue
    }

}

fun main() {
    val spreadsheet = Spreadsheet(3)
    spreadsheet.getValue("=5+7"); // returns 12 (5+7)
    spreadsheet.setCell("A1", 10); // sets A1 to 10
    spreadsheet.getValue("=A1+6"); // returns 16 (10+6)
    spreadsheet.setCell("B2", 15); // sets B2 to 15
    spreadsheet.getValue("=A1+B2"); // returns 25 (10+15)
    spreadsheet.resetCell("A1"); // resets A1 to 0
    spreadsheet.getValue("=A1+B2"); // returns 15 (0+15)
}