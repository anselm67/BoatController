package com.anselm.boatcontroller.controller

import android.util.Log
import com.anselm.boatcontroller.TAG
import kotlin.math.max
import kotlin.math.min

enum class MotorStatus(
    val value: Int,
    val label: String
) {
    Off(1, "Off"),
    Left(2, "Left"),
    Right(3, "Right"),
    Unknown(4, "Unknown");

    companion object {
        fun fromInt(value: Int) = entries.first { it.value == value }
    }
}

data class Status(
    val timestamp: Long,
    val isConnected: Boolean,
    val motorStatus: MotorStatus,
    val actualMotorStatus: MotorStatus,
    val ccValue: Int,
    val positionMm: Long,
) {
    companion object {
        val unknown = Status(
            System.currentTimeMillis(),
            false,
            MotorStatus.Unknown,
            MotorStatus.Unknown,
            -1,
            -1L,
        )
    }
}

abstract class Controller {

    var statusCallback: ((Status) -> Unit)? = null

    private var stopAfter = -1L
    private var thenCallback: ((Long) -> Unit)? = null

    protected fun armThenCallback(stopAfter: Long, callback: ((Long) -> Unit)?) {
        if (callback != null) {
            this.thenCallback?.invoke(lastStatus.positionMm)
            this.thenCallback = callback
            this.stopAfter = stopAfter
        }
    }

    protected var lastStatus: Status = Status(
        System.currentTimeMillis(),
        false,
        MotorStatus.Off,
        MotorStatus.Off,
        -1,
        0L
    )

    val positionMm
        get() = lastStatus.positionMm

    private fun isMotorRunning(): Boolean {
        return lastStatus.motorStatus == MotorStatus.Left
                || lastStatus.motorStatus == MotorStatus.Right
    }

    protected fun updateStatus(status: Status) {
        lastStatus = status
        try {
            statusCallback?.invoke(status)
        } catch (e: Exception) {
            Log.e(TAG, "updateStatus callback failed.", e)
        }
    }

    private var lastMotorStatusChangeTime = -1L

    protected fun handleProbeResult(motorStatus: MotorStatus, actualMotorStatus: MotorStatus) {
        val now = System.currentTimeMillis()
        val howLong = now - lastMotorStatusChangeTime
        var positionMm = lastStatus.positionMm

        // Updates the position.
        if (motorStatus != MotorStatus.Off) {
            val deltaMm = ((now - lastStatus.timestamp) * ARM_SPEED).toLong()
            positionMm = if (motorStatus == MotorStatus.Right) {
                min(ARM_LENGTH, positionMm + deltaMm)
            } else /* Left */ {
                max(0L, positionMm - deltaMm)
            }
        }

        if ( isMotorRunning() && stopAfter > 0 && howLong > stopAfter ) {
            Log.i(TAG, "Stopping howLong: $howLong, after: $stopAfter")
            stop()
            stopAfter = -1
        }

        if (lastStatus.motorStatus != motorStatus) {
            lastMotorStatusChangeTime = now

            // Run the callback - if any - only when the motor is stopped.
            if (motorStatus == MotorStatus.Off && thenCallback != null) {
                val callback = thenCallback
                this.thenCallback = null
                // The callback will issue more commands, and set a callback.
                try {
                    callback?.invoke(positionMm)
                } catch (e: Exception) {
                    Log.e(TAG, "Then callback failed.", e)
                }
            }
            Log.i(TAG, "Last command ran for $howLong ms, position $positionMm mm.")
        }

        if (motorStatus == MotorStatus.Off && actualMotorStatus == MotorStatus.Right) {
            positionMm = ARM_LENGTH
        } else if (motorStatus == MotorStatus.Off && actualMotorStatus == MotorStatus.Left) {
            positionMm = 0L
        }

        updateStatus(Status(now,
            true,
            motorStatus,
            actualMotorStatus,
            500,
            positionMm,
        ))
    }

    abstract fun calibrate()

    abstract fun connect()

    abstract fun disconnect()

    abstract fun left(forHowLong: Long = -1, callback: ((Long) -> Unit)? = null)

    abstract fun right(forHowLong: Long = -1, callback: ((Long) -> Unit)? = null)

    abstract fun stop()

    abstract fun goto(positionMm: Long, callback: ((Long) -> Unit)? = null)

    companion object {
        const val ARM_LENGTH = 300L
        // Arm speed in mm / ms
        const val ARM_SPEED = 0.00862
    }
}