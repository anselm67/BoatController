package com.anselm.boatcontroller.controller

import android.os.Handler
import android.os.Looper
import android.util.Log
import com.anselm.boatcontroller.TAG
import com.anselm.boatcontroller.controller.BLEController.Companion.PROBE_PERIOD_MILLIS

class ControllerMock: Controller() {
    private var connected = false
    private val handler = Handler(Looper.getMainLooper())

    private var motorStatus = MotorStatus.Off

    private fun probe() {
        handleProbeResult(motorStatus, motorStatus)
        handler.postDelayed(this::probe, this, PROBE_PERIOD_MILLIS)
    }

    override fun calibrate() {
        Log.d(TAG, "calibrate: nothing done.")
    }

    override fun connect() {
        if ( ! connected ) {
            Log.d(TAG, "connect")
            updateStatus(lastStatus.copy(
                timestamp = System.currentTimeMillis(),
                isConnected = true
            ))
            handler.postDelayed(this::probe, this, PROBE_PERIOD_MILLIS)
        }
        connected = true
    }

    override fun disconnect() {
        if ( connected ) {
            handler.removeCallbacksAndMessages(this)
            updateStatus(lastStatus.copy(
                timestamp = System.currentTimeMillis(),
                isConnected = false
            ))
        }
        connected = false
    }

    override fun left(forHowLong: Long, callback: ((Long) -> Unit)?) {
        armThenCallback(forHowLong, callback)
        motorStatus = MotorStatus.Left
    }

    override fun right(forHowLong: Long, callback: ((Long) -> Unit)?) {
        armThenCallback(forHowLong, callback)
        motorStatus = MotorStatus.Right
    }

    override fun stop() {
        motorStatus = MotorStatus.Off
    }

    override fun goto(positionMm: Long, callback: ((Long) -> Unit)?) {
        val runInMm = positionMm - lastStatus.positionMm
        val runInMs = (runInMm / ARM_SPEED).toLong()
        if ( runInMs > 0 ) {
            right(runInMs, callback)
        } else {
            left(- runInMs, callback)
        }
    }

}