package com.anselm.boatcontroller.controller

import android.os.Handler
import android.os.Looper
import android.util.Log
import com.anselm.boatcontroller.BuildConfig
import com.anselm.boatcontroller.TAG
import org.json.JSONObject

class BLEController : BLESocket.Callback, Controller() {

    private var socket: BLESocket? = null
    private val handler = Handler(Looper.getMainLooper())

    private var leftMinValue = CC_LEFT_MIN
    private var rightMaxValue = CC_RIGHT_MAX

    private fun send(cmd: ByteArray) {
        if (socket != null && socket!!.isConnected()) {
            try {
                socket!!.send(cmd)
            } catch (e: BLESocketException) {
                Log.e(TAG, "socket send failed on command $cmd", e)
            }
        }
    }

    private fun probe() {
        send(PROBE_BYTES)
    }

    private var armSpeed : Double = ARM_SPEED

    private fun parseProbe(data: String) {
        val obj = JSONObject(data)
        val actualMotorStatus = MotorStatus.fromInt(obj.getInt("M"))
        val ccValue = obj.getInt("C")
        if (actualMotorStatus == MotorStatus.Off) {
            leftMinValue = ccValue + 10
            rightMaxValue = ccValue - 10
        }
        val motorStatus = if (ccValue > leftMinValue) {
            MotorStatus.Left
        } else if (ccValue < rightMaxValue) {
            MotorStatus.Right
        } else {
            MotorStatus.Off
        }
        handleProbeResult(motorStatus, actualMotorStatus)

        handler.postDelayed(this::probe, this, PROBE_PERIOD_MILLIS)
    }

    override fun calibrate() {
        Log.i(TAG, "Calibration: right.")
        right {
            Log.i(TAG,"Calibration: left.")
            val start = System.currentTimeMillis()
            left {
                val end = System.currentTimeMillis()
                armSpeed = ARM_LENGTH / (end - start).toDouble()
                updateStatus(lastStatus.copy(
                    timestamp = System.currentTimeMillis(),
                    positionMm = 0L))
                Log.i(TAG, "Duration: ${end - start} ms, " +
                        "speed: ${String.format("%.2f",1000.0 * armSpeed)} mm/sec")
            }
        }
    }

    override fun connect() {
        if (socket == null ||  ! socket!!.isConnected()) {
            try{
                socket = BLESocket.create(DEVICE_ADDRESS, this)
            } catch (e: BLESocketException) {
                Log.e(TAG, "Failed to create socket to $DEVICE_ADDRESS")
            }
        }
    }

    override fun disconnect() {
        if ( socket != null ) {
            updateStatus(lastStatus.copy(
                timestamp = System.currentTimeMillis(),
                isConnected = false,
            ))
        }
        handler.removeCallbacksAndMessages(this)
        socket?.disconnect()
        socket = null
        updateStatus(lastStatus.copy(isConnected = false))
        Log.d(TAG, "Now disconnected.")
    }

    override fun left(forHowLong: Long, callback: ((Long) -> Unit)?) {
        armThenCallback(forHowLong, callback)
        send(LEFT_BYTES)
    }

    override fun right(forHowLong: Long, callback: ((Long) -> Unit)?) {
        armThenCallback(forHowLong, callback)
        send(RIGHT_BYTES)
    }

    override fun stop() {
        send(STOP_BYTES)
    }

    override fun goto(positionMm: Long, callback: ((Long) -> Unit)?) {
        val runInMm = positionMm - lastStatus.positionMm
        val runInMs = (runInMm / armSpeed).toLong()
        if ( runInMs > 0 ) {
            right(runInMs, callback)
        } else {
            left(- runInMs, callback)
        }
    }

    override fun onReceive(dataBytes: ByteArray) {
        val data = String(dataBytes)
        if (BuildConfig.DEBUG) {
            Log.d(TAG, "onReceive - invoked with $data")
        }
        parseProbe(data)
    }

    override fun onConnect() {
        Log.d(TAG, "onConnect() - invoked")
        updateStatus(lastStatus.copy(isConnected = true))
        handler.postDelayed(this::probe, this, PROBE_PERIOD_MILLIS)
    }

    override fun onDisconnect() {
        Log.d(TAG, "onDisconnect() - invoked")
        disconnect()
    }

    override fun onException(e: Exception) {
        Log.d(TAG, "onException() - invoked", e)
        disconnect()
    }

    companion object {
        const val DEVICE_ADDRESS: String = "9C:1D:58:A3:59:93"
        const val PROBE_PERIOD_MILLIS = 250L

        private const val CC_LEFT_MIN = 530
        private const val CC_RIGHT_MAX = 515

        private val PROBE_BYTES = byteArrayOf('P'.code.toByte())
        private val STOP_BYTES = byteArrayOf('S'.code.toByte())
        private val LEFT_BYTES: ByteArray = byteArrayOf('L'.code.toByte())
        private val RIGHT_BYTES: ByteArray = byteArrayOf('R'.code.toByte())
    }
}