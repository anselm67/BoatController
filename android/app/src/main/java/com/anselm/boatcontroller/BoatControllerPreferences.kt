package com.anselm.boatcontroller

import android.content.SharedPreferences

data class BoatControllerPreferences(
    var captureCount: Int,
    var analysisDelayMillis: Long,
    var alwaysOn: Boolean,
    var leftMount: Boolean,
) {
    fun save(prefs: SharedPreferences) {
        prefs.edit().apply {
            putInt("captureCount", captureCount)
            putLong("analysisDelayMillis", analysisDelayMillis)
            putBoolean("alwaysOn", alwaysOn)
            putBoolean("leftMount", leftMount)
        }.apply()
    }

    companion object {
        fun load(prefs: SharedPreferences) : BoatControllerPreferences{
            return BoatControllerPreferences(
                captureCount = prefs.getInt("captureCount", 5),
                analysisDelayMillis = prefs.getLong("analysisDelayMillis", 500L),
                alwaysOn = prefs.getBoolean("alwaysOn", true),
                leftMount = prefs.getBoolean("leftMount", true),
            )
        }
    }
}