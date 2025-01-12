package com.anselm.boatcontroller

import android.content.Context
import android.graphics.Bitmap
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.IOException
import java.nio.ByteBuffer
import kotlin.math.sqrt

class TensorflowClassifier(context: Context, medianCallback: (Double) -> Unit)
    : Classifier(medianCallback) {

    private var interpreter: Interpreter? = null
    private val outputTensor: TensorBuffer
    private val inputTensor: TensorBuffer


    @Throws(IOException::class)
    private fun initInterpreter(context: Context): Interpreter {
        val modelBytes = FileUtil.loadMappedFile(context, MODEL_FILENAME)
        val options = Interpreter.Options()
        options.setNumThreads(4)
        options.setUseNNAPI(false)
        return Interpreter(modelBytes, options)
    }

    init {
        // Initializes the model.
        interpreter = initInterpreter(context)
        outputTensor = TensorBuffer.createFixedSize(
            MODEL_OUTPUT_SHAPE,
            DataType.FLOAT32
        )
        inputTensor = TensorBuffer.createFixedSize(
            intArrayOf(
                INFERENCE_HEIGHT,
                INFERENCE_WIDTH,
                3
            ),
            DataType.FLOAT32
        )
    }

    private fun convertBitmapToByteBuffer(bitmap: Bitmap, tensor: TensorBuffer) {
        val byteBuffer = ByteBuffer.allocateDirect(bitmap.width * bitmap.height * 3 * 4) // 3 channels (RGB), 4 bytes per float
        byteBuffer.order(java.nio.ByteOrder.nativeOrder())

        val pixels = IntArray(bitmap.width * bitmap.height)
        bitmap.getPixels(pixels, 0, bitmap.width, 0, 0, bitmap.width, bitmap.height)

        val floatArray = FloatArray(bitmap.width * bitmap.height * 3)
        var mean = 0f

        for (i in pixels.indices) {
            val pixel = pixels[i]
            // Extract RGB channels from ARGB int
            val r = (pixel shr 16 and 0xFF).toFloat()
            val g = (pixel shr 8 and 0xFF).toFloat()
            val b = (pixel and 0xFF).toFloat()
            // Warning: this RGB order is meant to match the python code. Don't change it.
            floatArray[i * 3 + 2] = r
            floatArray[i * 3 + 1] = g
            floatArray[i * 3 ] = b
            mean += (r + g + b)
        }
        // Normalizes the image, the same way python does.
        mean /= floatArray.size.toFloat()
        val std = sqrt(floatArray.map { (it - mean) * (it - mean) }.average()).toFloat()
        for (i in floatArray.indices) {
            floatArray[i] = (floatArray[i] - mean) / std
        }
        tensor.loadArray(floatArray)
    }

    override fun infer(bitmap: Bitmap): FloatArray {
        convertBitmapToByteBuffer(bitmap, inputTensor)
        interpreter!!.run(inputTensor.buffer, outputTensor.buffer)
        return outputTensor.floatArray
    }


    companion object  {
        private const val MODEL_FILENAME = "banks.tflite"

    }
}