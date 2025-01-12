package com.anselm.boatcontroller.components

import android.graphics.Bitmap
import android.graphics.Matrix
import android.util.Log
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.viewinterop.AndroidView
import androidx.lifecycle.LifecycleOwner
import com.anselm.boatcontroller.BoatControllerApplication.Companion.app
import com.anselm.boatcontroller.models.ClassifierPreviewModel
import com.anselm.boatcontroller.models.LocalApplicationViewModel
import java.time.LocalDateTime
import java.time.format.DateTimeFormatter
import java.util.concurrent.Executors


private val YYYYMMDDFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd-HHmm")

private fun newBasename(): String {
    return LocalDateTime.now().format(YYYYMMDDFormatter)
}

private val ROTATE_90 by lazy {
    val matrix = Matrix()
    matrix.postRotate(90f)
    matrix
}

private class RateLimiter(private val intervalMillis: Long) {
    private var lastProcessedTime: Long = 0L

    fun throttle() {
        do {
            val currentTime = System.currentTimeMillis()
            val waitTime = lastProcessedTime + intervalMillis - currentTime
            if (waitTime <= 0) {
                lastProcessedTime = currentTime
                return
            } else {
                try {
                    Thread.sleep(waitTime)
                    return
                } catch (e: InterruptedException) { /* ignored and loop */ }
            }
        } while ( true )
    }
}

private fun process(
    viewModel: ClassifierPreviewModel,
    image: ImageProxy,
    rateLimiter: RateLimiter)
{
    rateLimiter.throttle()

    var basename: String? = null
    if (viewModel.captureCount > 0) {
        basename = newBasename()
        viewModel.captureCount -= 1
    }

    val bitmap = Bitmap.createBitmap(
        image.toBitmap(),
        0, 0,
        image.width, image.height,
        ROTATE_90,
        true
    )

    viewModel.updateTag(app.classifier.run(bitmap, basename))
}

@Composable
fun ClassifierPreview(viewModel: ClassifierPreviewModel) {
    val context = LocalContext.current
    val lifecycleOwner = LocalContext.current as LifecycleOwner
    val appViewModel = LocalApplicationViewModel.current
    val prefs by appViewModel.prefs.collectAsState()
    val cameraExecutor = remember { Executors.newSingleThreadExecutor() }
    val cameraProviderFuture = remember { ProcessCameraProvider.getInstance(context) }
    val previewView = remember { PreviewView(context).apply {
        scaleType = PreviewView.ScaleType.FILL_CENTER
        layoutParams = android.widget.FrameLayout.LayoutParams(
            android.view.ViewGroup.LayoutParams.MATCH_PARENT,
            android.view.ViewGroup.LayoutParams.MATCH_PARENT
        )
    }}


    LaunchedEffect(Unit) {
        val cameraProvider = cameraProviderFuture.get()
        val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
        val camera = cameraProvider.bindToLifecycle(lifecycleOwner, cameraSelector)
        val rateLimiter = RateLimiter(prefs.analysisDelayMillis)

        Log.d("ClassifierPreview", "sensorRotationDegrees: ${camera.cameraInfo.sensorRotationDegrees}" +
                " displayRotation: ${context.display.rotation}")


        val preview = Preview.Builder()
            .build().also {
                it.surfaceProvider = previewView.surfaceProvider
            }


        val imageAnalyzer = ImageAnalysis.Builder()
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
            .build()
            .also {
                it.setAnalyzer(cameraExecutor) { imageProxy ->
                    imageProxy.use { image ->  process(viewModel, image, rateLimiter) }
                }
            }

        try {
            cameraProvider.unbindAll()
            cameraProvider.bindToLifecycle(
                lifecycleOwner,
                cameraSelector,
                preview,
                imageAnalyzer
            )
        } catch (e: Exception) {
            Log.e("CameraX", "Binding failed", e)
        }

    }

    AndroidView(factory = { previewView }, modifier = Modifier.fillMaxSize())
}
