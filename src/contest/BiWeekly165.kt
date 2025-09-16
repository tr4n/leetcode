package contest

class BiWeekly165 {
    fun smallestAbsent(nums: IntArray): Int {
        val n = nums.size
        val set = nums.toSet()
        val sum = nums.sum()
        val avg = sum.toDouble() / n
        var ans = (avg.toInt() + 1).coerceAtLeast(1)
        while (ans in set) ans++
        return ans
    }

    fun minArrivalsToDiscard(arrivals: IntArray, w: Int, m: Int): Int {
        val n = arrivals.size
        val count = mutableMapOf<Int, Int>()
        val queue = ArrayDeque<Int>()
        var ans = 0

        for (i in 0 until n) {
            val item = arrivals[i]

            val outIndex = i - w
            if (outIndex >= 0 && queue.isNotEmpty() && queue.first() == outIndex) {
                val first = arrivals[outIndex]
                val c = count[first] ?: 0
                if (c <= 1) count.remove(first) else count[first] = c - 1
                queue.removeFirst()
            }
            //  println(queue)

            val cnt = count[item] ?: 0
            if (cnt < m) {
                count[item] = cnt + 1
                queue.add(i)
            } else {
                ans++
            }
        }
        return ans
    }

    fun generateSchedule(n: Int): Array<IntArray> {
        if (n <= 1 || n == 2 || n == 3) {
            return emptyArray()
        }

        val aMatches = mutableListOf<Pair<Int, Int>>()
        for (i in 0 until n - 1) {
            for (j in 0 until n - 1) {
                if (i != j) {
                    aMatches.add(i to j)
                }
            }
        }

        val bMatches = mutableListOf<Pair<Int, Int>>()
        for (i in 0 until n - 1) {
            bMatches.add(i to n - 1)
            bMatches.add(n - 1 to i)
        }

        val schedule = mutableListOf<IntArray>()
        var lastPlayedTeams = emptySet<Int>()

        for (day in 0 until n * (n - 1)) {
            val primaryGroup = if (day % 2 == 0) aMatches else bMatches
            val secondaryGroup = if (day % 2 == 0) bMatches else aMatches

            var foundMatch = false

            var matchIterator = primaryGroup.iterator()
            while (matchIterator.hasNext()) {
                val match = matchIterator.next()
                if (match.first !in lastPlayedTeams && match.second !in lastPlayedTeams) {
                    schedule.add(intArrayOf(match.first, match.second))
                    lastPlayedTeams = setOf(match.first, match.second)
                    matchIterator.remove()
                    foundMatch = true
                    break
                }
            }

            if (!foundMatch) {
                matchIterator = secondaryGroup.iterator()
                while (matchIterator.hasNext()) {
                    val match = matchIterator.next()
                    if (match.first !in lastPlayedTeams && match.second !in lastPlayedTeams) {
                        schedule.add(intArrayOf(match.first, match.second))
                        lastPlayedTeams = setOf(match.first, match.second)
                        matchIterator.remove()
                        break
                    }
                }
            }
        }

        return schedule.toTypedArray()
    }
}

fun generateSchedule1(n: Int): Array<IntArray> {
    if (n <= 1 || n == 2 || n == 3) {
        return emptyArray()
    }


    val firstHalfMatches = mutableSetOf<Pair<Int, Int>>()
    for (i in 0 until n) {
        for (j in i + 1 until n) {
            firstHalfMatches.add(i to j)
        }
    }
    val secondHalfMatches = firstHalfMatches.map { (h, a) -> a to h }.toMutableSet()

    val schedule = mutableListOf<IntArray>()
    val totalMatches = n * (n - 1)

    fun solve(lastPlayedTeams: Set<Int>): Boolean {
        if (schedule.size == totalMatches) {
            return true
        }


        val currentMatches = if (schedule.size < totalMatches / 2) {
            firstHalfMatches
        } else {
            secondHalfMatches
        }

        val possibleMatches = currentMatches.filter { (home, away) ->
            home !in lastPlayedTeams && away !in lastPlayedTeams
        }

        for (match in possibleMatches) {
            val (home, away) = match


            schedule.add(intArrayOf(home, away))
            currentMatches.remove(match)

            if (solve(setOf(home, away))) {
                return true
            }


            currentMatches.add(match)
            schedule.removeAt(schedule.lastIndex)
        }

        return false
    }

    solve(emptySet())
    return schedule.toTypedArray()
}

fun generateSchedule(n: Int): Array<IntArray> {
    if (n <=3) return emptyArray()

    val teams = mutableListOf<Int>()
    for (i in 0 until n) {
        teams.add(i)
    }
    val byeTeam = if (n % 2 != 0) {
        teams.add(n)
        n
    } else -1
    val nTeams = teams.size
    val halfN = nTeams / 2

    val schedules = mutableListOf<IntArray>()

    repeat(nTeams - 1) {
        val roundMatches = mutableListOf<IntArray>()
        for (i in 0 until halfN) {
            val home = teams[i]
            val away = teams[nTeams - 1 - i]
            if (home != byeTeam && away != byeTeam) {
                roundMatches.add(intArrayOf(home, away))
            }
        }
        schedules.addAll(roundMatches)

        val first = teams[0]
        val last = teams.removeAt(nTeams - 1)
        teams.add(1, last)
        teams[0] = first
    }

    val secondSchedules = schedules.map { intArrayOf(it[1], it[0]) }
    schedules.addAll(secondSchedules)

    return schedules.toTypedArray()
}

fun main() {
    val contest = BiWeekly165()

    println(
        contest.minArrivalsToDiscard(
            intArrayOf(7, 3, 9, 9, 7, 3, 5, 9, 7, 2, 6, 10, 9, 7, 9, 1, 3, 6, 2, 4, 6, 2, 6, 8, 4, 8, 2, 7, 5, 6),
            10,
            1
        )
    )
}