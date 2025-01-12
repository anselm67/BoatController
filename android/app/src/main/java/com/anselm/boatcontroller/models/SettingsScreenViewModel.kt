package com.anselm.boatcontroller.models

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import com.anselm.boatcontroller.BoatControllerApplication.Companion.app

class SettingsScreenViewModel : ViewModel() {
    var captureCount by mutableIntStateOf(app.prefs.captureCount)
    var analysisDelayMillis by mutableLongStateOf(app.prefs.analysisDelayMillis)
    var alwaysOn by mutableStateOf(app.prefs.alwaysOn)
    var armIndex by mutableIntStateOf(if (app.prefs.leftMount) 0 else 1)

    // Progress report for exporting images and labels.
    var showProgress by mutableStateOf(false)
    var progress by mutableFloatStateOf(0f)
}