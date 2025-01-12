package com.anselm.boatcontroller.models

import androidx.compose.runtime.compositionLocalOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import com.anselm.boatcontroller.BoatControllerApplication.Companion.app
import com.anselm.boatcontroller.BoatControllerPreferences
import com.anselm.boatcontroller.controller.Status
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow

val LocalApplicationViewModel = compositionLocalOf<ApplicationViewModel> {
    error("No ApplicationViewModel provided.")
}

data class ApplicationState(
    val hideBottomBar: Boolean = false,
    val title: String = "Boat Controller",
    val classifierEnabled: Boolean = true,
    val autoPilotEnabled: Boolean = false,
)

class ApplicationViewModel : ViewModel() {
    private val statusFlow = MutableStateFlow(Status.unknown)
    val status = statusFlow.asStateFlow()

    fun updateStatus(status: Status) {
        statusFlow.value = status
    }


    private val applicationStateFlow = MutableStateFlow(ApplicationState())
    val applicationState = applicationStateFlow.asStateFlow()

    fun updateApplicationState(change: (state: ApplicationState) -> ApplicationState): ApplicationViewModel {
        val oldState = applicationStateFlow.value
        var newState = change(oldState)
        if ( ! newState.classifierEnabled && newState.autoPilotEnabled ) {
            newState = newState.copy(autoPilotEnabled = false)
        }
        // Performs any corresponding app updates:
        if (newState.classifierEnabled != oldState.classifierEnabled) {
            app.classifierEnabled(newState.classifierEnabled)
        }
        if (newState.autoPilotEnabled != oldState.autoPilotEnabled) {
            app.autoPilotEnabled(newState.autoPilotEnabled)
        }
        applicationStateFlow.value = newState
        return this
    }

    private val prefsFlow = MutableStateFlow(app.prefs)
    val prefs = prefsFlow.asStateFlow()

    fun updatePreferences(change: (BoatControllerPreferences) -> BoatControllerPreferences) {
        prefsFlow.value = app.updatePreferences(change)
    }


    class Factory :
        ViewModelProvider.NewInstanceFactory() {
        override fun <T : ViewModel> create(modelClass: Class<T>): T {
            @Suppress("UNCHECKED_CAST")
            return requireNotNull(value = ApplicationViewModel() as? T) {
                "Cannot create an instance of $modelClass"
            }
        }
    }

}