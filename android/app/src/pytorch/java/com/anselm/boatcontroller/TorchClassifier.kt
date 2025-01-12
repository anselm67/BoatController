package com.anselm.boatcontroller

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.pytorch.IValue
import org.pytorch.LitePyTorchAndroid
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils


class TorchClassifier(context: Context, medianCallback: (Double) -> Unit)
    : Classifier(medianCallback)
{
    private var module: Module? = null

    private fun initModule(context: Context): Module? {
        Log.d(TAG, "initModule!")
        var module: Module? = null
        try {
            module = LitePyTorchAndroid.loadModuleFromAsset(context.assets, MODULE_NAME)
            Log.d(TAG, "Loaded module from $MODULE_NAME")
        } catch (e: Exception) {
            Log.e(TAG, "Unable to load module $MODULE_NAME", e)
        }
        return module
    }

    init {
        module = initModule(context)
    }

    override fun infer(bitmap: Bitmap): FloatArray {
        val inputBitmap = Bitmap.createScaledBitmap(bitmap, INFERENCE_WIDTH, INFERENCE_HEIGHT, true)
        val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
            inputBitmap,
            floatArrayOf(94.9309f / 255f, 95.9795f / 255f, 77.9645f / 255f),     // means
            floatArrayOf(53.8035f / 255f, 54.9085f / 255f, 60.3234f / 255f)      // std
        )
        val output = module!!.forward(IValue.from(inputTensor))
        val outputTensor = output.toTensor()
        return outputTensor.dataAsFloatArray
    }

    companion object {
        private const val MODULE_NAME = "banks.ptl"
    }
}