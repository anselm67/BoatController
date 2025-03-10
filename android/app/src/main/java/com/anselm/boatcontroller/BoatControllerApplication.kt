package com.anselm.boatcontroller

import android.Manifest
import android.app.Application
import android.content.Context
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.util.Log
import android.widget.Toast
import androidx.core.app.ActivityCompat
import com.anselm.boatcontroller.controller.BLEController
import com.anselm.boatcontroller.controller.Controller
import com.anselm.boatcontroller.controller.Controller.Companion.ARM_LENGTH
import com.anselm.boatcontroller.controller.ControllerMock
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.launch
import java.io.BufferedOutputStream
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.math.abs

class BoatControllerApplication: Application() {
    lateinit var controller: Controller
    lateinit var classifier: Classifier
    private lateinit var captureDirectory: File
    lateinit var prefs: BoatControllerPreferences
    val applicationScope = CoroutineScope(SupervisorJob())

    private fun initClassifier(className: String): Classifier {
        return Class.forName(className)
            .getDeclaredConstructor(Context::class.java, Function1::class.java)
            .newInstance(this, { it: Double -> steer(it) }) as Classifier
    }

    override fun onCreate() {
        super.onCreate()
        app = this
        prefs = loadPreferences()
        controller = if ( prefs.useMockController ) ControllerMock() else BLEController()
        // Initiates the classifier.
        @Suppress("KotlinConstantConditions")
        val className = when(BuildConfig.FLAVOR) {
            "pytorch" -> "com.anselm.boatcontroller.TorchClassifier"
            "tensorflow" -> "com.anselm.boatcontroller.TensorflowClassifier"
            else -> throw NotImplementedError("${BuildConfig.FLAVOR} has no classifier class defined.")
        }
        try {
            classifier = initClassifier(className)
        } catch (e: Exception) {
            Log.e(TAG, "Unable to initiate classifier $className", e)
        }
        Log.i(TAG, "Classifier of ${classifier.javaClass::class.qualifiedName} built.")
        // Initiates images and labels capture.
        captureDirectory = File(applicationContext!!.filesDir, "capture")
        captureDirectory.mkdirs()
    }

    private fun loadPreferences(): BoatControllerPreferences {
        return BoatControllerPreferences.load(applicationContext.getSharedPreferences(
            "preferences", MODE_PRIVATE
        ))
    }

    fun updatePreferences(change: (BoatControllerPreferences) -> BoatControllerPreferences)
        : BoatControllerPreferences
    {
        val useMockController = prefs.useMockController
        prefs = change(prefs)
        prefs.save(applicationContext.getSharedPreferences("preferences", MODE_PRIVATE))
        if ( prefs.useMockController != useMockController ) {
            controller.disconnect()
            if ( prefs.useMockController ) {
                Log.i(TAG, "Switching mock controller to MokController")
                controller = ControllerMock()
            } else {
                Log.i(TAG, "Switching mock controller to BLEController")
                controller = BLEController()
            }

        }
        return prefs
    }

    fun toast(msg: String) {
        applicationScope.launch(Dispatchers.Main) {
            Toast.makeText(applicationContext, msg, Toast.LENGTH_LONG).show()
        }
    }


    val allPermissions = arrayOf(
        Manifest.permission.BLUETOOTH_SCAN,
        Manifest.permission.BLUETOOTH_CONNECT,
        Manifest.permission.BLUETOOTH_ADVERTISE,
        Manifest.permission.ACCESS_COARSE_LOCATION,
        Manifest.permission.ACCESS_FINE_LOCATION,
        Manifest.permission.CAMERA
    )

    fun checkPermissions(): Boolean {
        return allPermissions.all {
            ActivityCompat.checkSelfPermission(
                this,
                it
            ) == PackageManager.PERMISSION_GRANTED
        }
    }

    private val averager = Averager(windowSize = 5)
    private var autoPilotOn = false

    fun classifierEnabled(onOff: Boolean) {
        classifier.enable(onOff)
        if ( ! onOff ) {
            autoPilotEnabled(false)
        }
    }

    fun autoPilotEnabled(onOff:Boolean) {
        averager.reset()
        autoPilotOn = onOff
    }

    private fun steer(median: Double) {
        if ( ! autoPilotOn ) {
            return
        }
        averager.append(median)
        var newPosition = (ARM_LENGTH * averager.average()).toLong()
        if ( ! prefs.leftMount ) {
            newPosition = ARM_LENGTH - newPosition
        }
        if (abs(newPosition - controller.positionMm) / ARM_LENGTH.toDouble() > 0.1) {
            Log.d(TAG, "Steering to $median, request $newPosition mm")
            controller.goto(newPosition) {
                Log.d(TAG, "Steering completed or aborted $it mm.")
            }
        } else {
            Log.d(TAG, "Steering to $median, request $newPosition mm (within scope).")
        }
    }

    fun saveBitmap(bm: Bitmap, filename: String): Boolean {
        try {
            val file = File(captureDirectory, filename)
            FileOutputStream(file).use { outputStream ->
                BufferedOutputStream(outputStream).use {
                    bm.compress(Bitmap.CompressFormat.PNG, 99, it)
                }
            }
            return true
        } catch (e: IOException) {
            Log.e(TAG, "Failed to save bitmap to $filename")
        }
        return false
    }

    fun deleteCapturedFiles() {
        captureDirectory.listFiles()?.forEach { file ->
            if ( file.isFile ) {
                file.delete()
            }
        }
    }

    fun listCaptureFiles() : Array<File> {
        return captureDirectory.listFiles() ?: emptyArray()
    }

    val captureFileCount: Int
        get() {
            return captureDirectory.listFiles()?.size ?: 0
        }

    companion object {
        lateinit var app: BoatControllerApplication
            private set
    }
}