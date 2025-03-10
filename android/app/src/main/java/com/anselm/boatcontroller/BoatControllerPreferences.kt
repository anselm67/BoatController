package com.anselm.boatcontroller

import android.content.SharedPreferences
import androidx.core.content.edit

data class BoatControllerPreferences(
    var captureCount: Int,
    var analysisDelayMillis: Long,
    var alwaysOn: Boolean,
    var leftMount: Boolean,
    var useMockController: Boolean,
) {
    fun save(prefs: SharedPreferences) {
        prefs.edit() {
            putInt("captureCount", captureCount)
            putLong("analysisDelayMillis", analysisDelayMillis)
            putBoolean("alwaysOn", alwaysOn)
            putBoolean("leftMount", leftMount)
            putBoolean("useMockController", useMockController)
        }
    }

    companion object {
        fun load(prefs: SharedPreferences) : BoatControllerPreferences{
            return BoatControllerPreferences(
                captureCount = prefs.getInt("captureCount", 5),
                analysisDelayMillis = prefs.getLong("analysisDelayMillis", 500L),
                alwaysOn = prefs.getBoolean("alwaysOn", true),
                leftMount = prefs.getBoolean("leftMount", true),
                useMockController = prefs.getBoolean("useMockController", false),
            )
        }
    }
}