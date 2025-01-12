package com.anselm.boatcontroller

class Averager(private val windowSize: Int) {
    private val values = ArrayDeque<Double>()
    private var total = 0.0

    init {
        require(windowSize > 0) { "Window size must be greater than 0." }
    }

    fun append(value: Double) {
        if (values.size == windowSize) {
            total -= values.removeFirst() // Remove the oldest value from the total
        }
        values.addLast(value)
        total += value
    }

    fun average(): Double {
        if (values.isEmpty()) throw IllegalStateException("No values added yet.")
        return total / values.size
    }

    fun reset() {
        values.clear()
    }
}