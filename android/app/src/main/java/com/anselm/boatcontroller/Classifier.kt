package com.anselm.boatcontroller

import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.Log
import androidx.annotation.OptIn
import androidx.camera.core.ExperimentalGetImage
import com.anselm.boatcontroller.BoatControllerApplication.Companion.app

abstract class Classifier(
    private val medianCallback: (Double) -> Unit
) {
    private var inferenceCount = 0
    private var enabled: Boolean = true

    /**
     * The median width of the amount of blue / water in the classifier output.
     * This is normalized between 0.0 and 1.0
     */
    private var median = 0.0


    private fun updateMedian(output: IntArray) {
        val water = IntArray(INFERENCE_WIDTH)
        var total = 0

        for (y in 0 until INFERENCE_HEIGHT) {
            for (x in 0 until INFERENCE_WIDTH) {
                if (output[y * INFERENCE_WIDTH + x] == BLUE) {
                    water[x]++
                    total++
                }
            }
        }

        var median = 0
        this.median = 1.0
        for (x in 0 until INFERENCE_WIDTH) {
            median += water[x]
            if (median > total / 2) {
                this.median = (x.toDouble()) / INFERENCE_WIDTH.toDouble()
                break
            }
        }
        medianCallback.invoke(this.median)
    }

    private fun cropAndScaleForInference(inputBitmap: Bitmap): Bitmap {
        val targetHeight = (inputBitmap.width.toFloat() / ASPECT_RATIO).toInt()
        val crop = Bitmap.createBitmap(
            inputBitmap,
            0, (inputBitmap.height - targetHeight) / 2,
            inputBitmap.width, targetHeight)
        return Bitmap.createScaledBitmap(
            crop,
            INFERENCE_WIDTH, INFERENCE_HEIGHT,
            true
        )

    }

    private fun drawMedian(bitmap: Bitmap): Bitmap {
        val outputBitmap = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(outputBitmap)

        val colorPaint = { lineColor: Int ->
            Paint().apply {
                color = lineColor
                strokeWidth = 3f
            }
        }
        for (line in listOf(Pair(0.5, colorPaint(Color.WHITE)), Pair(median, colorPaint(Color.BLUE)))) {
            val (x, paint) = line
            canvas.drawLine(
                (x * outputBitmap.width).toFloat(), 0f,
                (x * outputBitmap.width).toFloat(),
                bitmap.height.toFloat(),
                paint
            )
        }
        return outputBitmap
    }


    fun enable(onOff: Boolean) {
        enabled = onOff
    }

    abstract fun infer(bitmap: Bitmap): FloatArray

    @OptIn(ExperimentalGetImage::class)
    fun run(bitmap: Bitmap, basename: String? = null): Pair<Double, Bitmap>? {
        val inputBitmap = cropAndScaleForInference(bitmap)
        if (basename != null) {
            app.saveBitmap(
                inputBitmap,
                "$basename-input-$inferenceCount.png"
            )
        }
        if ( ! enabled ) {
            return null
        }
        val startTime = System.currentTimeMillis()
        val values = infer(inputBitmap)
        val endTime = System.currentTimeMillis()

        val size = INFERENCE_HEIGHT * INFERENCE_WIDTH
        val imgtag = IntArray(size)
        for (i in 0..<size) {
            imgtag[i] = if (values[i] > 0.5) GREEN else BLUE
        }
        updateMedian(imgtag)
        Log.i("com.anselm.boatcontroller.Inference",
            "Inference ran in ${endTime-startTime} ms, median: ${median}.")

        val outputBitmap = Bitmap.createBitmap(
            imgtag,
            INFERENCE_WIDTH,
            INFERENCE_HEIGHT,
            Bitmap.Config.ARGB_8888
        )

        if (basename != null) {
            app.saveBitmap(
                outputBitmap,
                "${basename}-output-${inferenceCount}.png"
            )
        }

        inferenceCount++
        return Pair(median, drawMedian(outputBitmap))
    }

    companion object {
        private const val BLUE = (0xff) shl 24 or (0xff)
        private const val GREEN = (0xff) shl 24 or ((0xff) shl 8)

        const val INFERENCE_WIDTH: Int = 320
        const val INFERENCE_HEIGHT: Int = 180
        const val ASPECT_RATIO = INFERENCE_WIDTH.toFloat() / INFERENCE_HEIGHT.toFloat()

        val MODEL_OUTPUT_SHAPE = intArrayOf(1, 320, 180, 2)
    }
}