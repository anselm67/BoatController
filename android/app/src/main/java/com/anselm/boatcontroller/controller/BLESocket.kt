package com.anselm.boatcontroller.controller

import android.annotation.SuppressLint
import android.bluetooth.BluetoothDevice
import android.bluetooth.BluetoothGatt
import android.bluetooth.BluetoothGattCallback
import android.bluetooth.BluetoothGattCharacteristic
import android.bluetooth.BluetoothGattDescriptor
import android.bluetooth.BluetoothManager
import android.bluetooth.BluetoothStatusCodes
import android.bluetooth.le.BluetoothLeScanner
import android.bluetooth.le.ScanCallback
import android.bluetooth.le.ScanResult
import android.content.Context.BLUETOOTH_SERVICE
import android.os.Handler
import android.os.Looper
import android.util.Log
import com.anselm.boatcontroller.BoatControllerApplication.Companion.app
import com.anselm.boatcontroller.TAG
import java.util.UUID
import java.util.concurrent.Semaphore
import java.util.concurrent.TimeUnit

class BLESocketException(message: String) : Exception(message)

class BLESocket(
    private val deviceAddress: String,
    callback: Callback,
) {
    private val bluetoothAdapter: BluetoothManager = app.getSystemService(BLUETOOTH_SERVICE) as BluetoothManager
    private val callbacks = mutableListOf<Callback>()
    private var gatt: BluetoothGatt? = null
    private var rx: BluetoothGattCharacteristic? = null
    private var tx:BluetoothGattCharacteristic? = null

    init {
        callbacks.add(callback)
    }

    @SuppressLint("MissingPermission")
    fun connect(device: BluetoothDevice) {
        Log.i(TAG, "connect $device @ ${device.address}")
        gatt = device.connectGatt(app, false, object : BluetoothGattCallback() {

            override fun onConnectionStateChange(gatt: BluetoothGatt, status: Int, newState: Int) {
                super.onConnectionStateChange(gatt, status, newState)
                if (newState == BluetoothGatt.STATE_CONNECTED) {
                    if (status == BluetoothGatt.GATT_SUCCESS) {
                        // Connected to device, start discovering services to finalize the connection.
                        if (!gatt.discoverServices()) {
                            disconnect()
                            doException("Failed to discover services.")
                        }
                    } else {
                        disconnect()
                        callbacks.forEach {
                            it.onException(BLESocketException("Connection state changed to newState $newState, with status $status"))
                        }
                    }
                } else if (newState == BluetoothGatt.STATE_DISCONNECTED) {
                    // Disconnected, notify callbacks of disconnection.
                    rx = null
                    tx = null
                    doDisconnect()
                }
            }

            override fun onServicesDiscovered(gatt: BluetoothGatt, status: Int) {
                super.onServicesDiscovered(gatt, status)
                // Checks that we were able to discover the device's services.
                if (status != BluetoothGatt.GATT_SUCCESS) {
                    disconnect()
                    doException("onServicesDiscovered failed with status $status")
                    return
                }
                // Checks that the serial service is available.
                val serial = gatt.getService(SERIAL_SERVICE_UUID)
                if (serial == null) {
                    disconnect()
                    doException("onServicesDiscovered: no service for $SERIAL_SERVICE_UUID")
                    return
                }
                // Gets the RX/TX characteristics to exchange data and enable notifications on RX.
                tx = gatt.getService(SERIAL_SERVICE_UUID).getCharacteristic(TX_CHAR_UUID)
                rx = gatt.getService(SERIAL_SERVICE_UUID).getCharacteristic(RX_CHAR_UUID)
                if ( rx == null || ! enableNotifications(rx!!) ) {
                    disconnect()
                    doException("Failed to enable notifications on RX")
                }
            }

            override fun onCharacteristicWrite(
                gatt: BluetoothGatt,
                characteristic: BluetoothGattCharacteristic,
                status: Int
            ) {
                super.onCharacteristicWrite(gatt, characteristic, status)
                unlock()
                if (status != BluetoothGatt.GATT_SUCCESS) {
                    doException("characteristicWrite failed status $status")
                }
            }

            override fun onCharacteristicChanged(
                gatt: BluetoothGatt,
                characteristic: BluetoothGattCharacteristic,
                value: ByteArray
            ) {
                super.onCharacteristicChanged(gatt, characteristic, value)
                doReceive(value)
            }

            override fun onDescriptorWrite(
                gatt: BluetoothGatt,
                descriptor: BluetoothGattDescriptor,
                status: Int
            ) {
                super.onDescriptorWrite(gatt, descriptor, status)
                unlock()
                if (status == BluetoothGatt.GATT_SUCCESS) {
                    // Whooohooo we're connected, at last.
                    doConnect()
                } else {
                    disconnect()
                    doException("Failed to enable RX notifications, status $status")
                }
            }
        })
    }

    @SuppressLint("MissingPermission")
    private fun enableNotifications(c: BluetoothGattCharacteristic): Boolean {
        // Enables the notifications on that characteristic.
        // We double check the gatt connection as it may have been closed in the meantime.
        if (gatt == null || !gatt!!.setCharacteristicNotification(c, true)) {
            Log.i(TAG, "Failed to set characteristic ${c.uuid} notifications.")
            return false
        }
        // Writes out its descriptor.
        val descriptor = c.getDescriptor(CLIENT_UUID)
        if (descriptor == null) {
            Log.i(TAG, "Failed to get client $CLIENT_UUID descriptor.")
            return false
        }

        val enableNotificationValue = BluetoothGattDescriptor.ENABLE_NOTIFICATION_VALUE
        if (lock { gatt!!.writeDescriptor(descriptor, enableNotificationValue) != BluetoothStatusCodes.SUCCESS}) {
            unlock()
            Log.i(TAG, "Failed to write characteristic ${descriptor.uuid} descriptor (ignore).")
            // This fails unreliably, but doesn't prevent us from getting notified, so
            // we keep going instead of returning false (which would trigger a full disconnect).
        }
        // The 'writing' lock will be released when onDescriptionWrite fires.
        return true
    }

    private val writeSemaphore = Semaphore(1, true)

    private fun lock(f: (BLESocket) -> Boolean): Boolean {
        if ( writeSemaphore.tryAcquire(WRITE_LOCK_TIMEOUT_MILLIS, TimeUnit.MILLISECONDS) ) {
            return f(this)
        } else {
            disconnect()
            doException("Semaphore wait timed out.")
            return false
        }
    }

    private fun unlock() {
        writeSemaphore.release()
    }

    @SuppressLint("MissingPermission")
    fun disconnect() {
        if (gatt != null) {
            callbacks.clear()
            gatt!!.disconnect()
            gatt!!.close()
        }
        gatt = null
        rx = null
        tx = null
    }

    private fun doException(errorMessage: String) {
        callbacks.forEach {
            try {
                it.onException(BLESocketException(errorMessage))
            } catch (e: Exception) {
                Log.e(TAG, "Callback onException() failed (ignored).", e)
            }
        }
    }

    private fun doDisconnect() {
        callbacks.forEach {
            try {
                it.onDisconnect()
            } catch (e: Exception) {
                Log.e(TAG, "Callback disconnect() failed (ignored).", e)
            }
        }
    }

    private fun doConnect() {
        callbacks.forEach {
            try {
                it.onConnect()
            } catch (e: Exception) {
                Log.e(TAG, "Callback onConnect() failed (ignored).", e)
            }
        }
    }

    private fun doReceive(value: ByteArray) {
        callbacks.forEach {
            try {
                it.onReceive(value)
            } catch (e: Exception) {
                Log.e(TAG, "Callback onReceive() failed (ignored).", e)
            }
        }
    }

/*
    private fun startScan() {
        Log.i(TAG, "startScan")
        val bluetoothDevice: BluetoothDevice = bluetoothAdapter.adapter.getRemoteDevice(deviceAddress)
        connect(bluetoothDevice)
    }
*/




    var scanner: BluetoothLeScanner? = null
    private val scanCallback = object : ScanCallback() {
        @SuppressLint("MissingPermission")
        override fun onScanResult(callbackType: Int, result: ScanResult) {
            if (result.device.address == deviceAddress) {
                println("Device found! Connecting...")
                scanner?.stopScan(this)
                connect(result.device)
            }
            scanner = null
        }

        @SuppressLint("MissingPermission")
        override fun onScanFailed(errorCode: Int) {
            super.onScanFailed(errorCode)
            scanner?.stopScan(this)
            val errorMessage = when (errorCode) {
                SCAN_FAILED_ALREADY_STARTED -> "scan already started"
                SCAN_FAILED_APPLICATION_REGISTRATION_FAILED -> "application registration failed"
                SCAN_FAILED_FEATURE_UNSUPPORTED -> "feature unsupported"
                SCAN_FAILED_INTERNAL_ERROR -> "internal error"
                else -> "unknown error"
            }
            scanner = null
            doException("BLE scan failed, error $errorMessage ($errorCode)")
        }
    }

    @SuppressLint("MissingPermission")
    fun startScan() {
        scanner = bluetoothAdapter.adapter.bluetoothLeScanner
        scanner!!.startScan(scanCallback)
        Handler(Looper.getMainLooper()).postDelayed({
            Log.d(TAG, "Scanning timeout: ok? ${scanner != null}")
            if (scanner != null) {
                scanner!!.stopScan(scanCallback)
                doException("BLE scan timed out.")
            }
        }, 5000L) // Scan for 10 seconds
    }

    @SuppressLint("MissingPermission")
    fun send(dataBytes: ByteArray) {
        if (tx == null) {
            throw BLESocketException("send: no TX characteristic.")
        }
        if (lock {
                gatt!!.writeCharacteristic(
                    tx!!,
                    dataBytes,
                    BluetoothGattCharacteristic.WRITE_TYPE_DEFAULT
                ) != BluetoothStatusCodes.SUCCESS
            }) {
            unlock()
            throw BLESocketException("writeCharacteristic failed.")
        }
        // We'll unlock upon receiving the write confirmation in our gatt callback.
    }

    fun isConnected(): Boolean {
        return tx != null && rx != null
    }

    interface Callback {
        /**
         * The devices has transmitted data,
         * @param dataBytes The data transmitted by the device.
         */
        fun onReceive(dataBytes: ByteArray)

        /**
         * The connection has been established with the device.
         */
        fun onConnect()

        /**
         * The device was disconnected.
         */
        fun onDisconnect()

        /**
         * An error occurred while interacting with the device.
         * @param e The relevant exception.
         */
        fun onException(e: Exception)
    }

    companion object {
        private val CLIENT_UUID = UUID.fromString("00002902-0000-1000-8000-00805f9b34fb" /*""0000FFE1-0000-1000-8000-00805F9B34FB" */)
        private val SERIAL_SERVICE_UUID = UUID.fromString("0000ffe0-0000-1000-8000-00805f9b34fb")
        private val TX_CHAR_UUID = UUID.fromString("0000ffe1-0000-1000-8000-00805f9b34fb")
        private val RX_CHAR_UUID = UUID.fromString("0000ffe1-0000-1000-8000-00805f9b34fb")

        const val WRITE_LOCK_TIMEOUT_MILLIS = 50L
        fun create(deviceAddress: String, callback: Callback): BLESocket {
            val socket = BLESocket(deviceAddress, callback)
            socket.startScan()
            return socket
        }
    }
}