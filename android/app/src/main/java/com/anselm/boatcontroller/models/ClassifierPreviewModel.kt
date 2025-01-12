package com.anselm.boatcontroller.models

import android.graphics.Bitmap
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import kotlinx.coroutines.flow.MutableStateFlow

class ClassifierPreviewModel: ViewModel() {
    val tagFlow: MutableStateFlow<Pair<Double, Bitmap>?> = MutableStateFlow(null)
    fun updateTag(value: Pair<Double, Bitmap>?) {
        tagFlow.value = value
    }

    var captureCount by mutableIntStateOf(0)
}