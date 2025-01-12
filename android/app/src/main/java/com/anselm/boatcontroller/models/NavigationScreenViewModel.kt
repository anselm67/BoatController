package com.anselm.boatcontroller.models

import androidx.lifecycle.ViewModel

class NavigationScreenViewModel(
    val classifierPreviewModel: ClassifierPreviewModel = ClassifierPreviewModel()
): ViewModel() {

    var captureCount: Int
        get() = classifierPreviewModel.captureCount
        set(value) {
            classifierPreviewModel.captureCount = value
        }
}