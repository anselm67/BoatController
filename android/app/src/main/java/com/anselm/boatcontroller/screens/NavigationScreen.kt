package com.anselm.boatcontroller.screens

import android.util.Log
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableLongStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.alpha
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.zIndex
import androidx.lifecycle.viewmodel.compose.viewModel
import com.anselm.boatcontroller.BoatControllerApplication.Companion.app
import com.anselm.boatcontroller.Classifier
import com.anselm.boatcontroller.R
import com.anselm.boatcontroller.TAG
import com.anselm.boatcontroller.components.AppButton
import com.anselm.boatcontroller.components.ClassifierPreview
import com.anselm.boatcontroller.controller.Controller.Companion.ARM_LENGTH
import com.anselm.boatcontroller.controller.Status
import com.anselm.boatcontroller.models.LocalApplicationViewModel
import com.anselm.boatcontroller.models.NavigationScreenViewModel


@Composable
fun Controls() {
    val appViewModel = LocalApplicationViewModel.current
    val status by appViewModel.status.collectAsState()
    val state by appViewModel.applicationState.collectAsState()

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(10.dp)
            .fillMaxWidth(),
        horizontalArrangement = Arrangement.SpaceAround,
    ) {
        AppButton(onClick = {
            appViewModel.updateApplicationState { it.copy(
                classifierEnabled = ! it.classifierEnabled
            ) }
        }) {
            Text(text = if (state.classifierEnabled) "Hide Tags" else "Show Tags")
        }
        AppButton(
            onClick = {
                appViewModel.updateApplicationState { it.copy(
                    autoPilotEnabled = ! it.autoPilotEnabled
                ) }
            }, enabled = state.classifierEnabled && status.isConnected
        ) {
            Text(if (state.autoPilotEnabled) "Disable Auto-Pilot" else "Enable Auto-Pilot")
        }
        AppButton(onClick = {
            Log.d(TAG, "Disconnecting ...")
            app.controller.disconnect()
        }, enabled = status.isConnected) {
            Text(text = "Disconnect")
        }
    }
}

@Composable
private fun LabelValue(modifier: Modifier, label: String, value: String) {
    Row(
        modifier = modifier,
        horizontalArrangement = Arrangement.SpaceBetween) {
        Text(text = label, fontWeight = FontWeight.Bold)
        Text(text = value)
    }
}

@Composable
private fun LabelValue(label: String, value: String) {
    LabelValue(Modifier.fillMaxWidth(), label, value)
}

@Composable
fun Status(status: Status) {
    Column(
        modifier = Modifier
            .padding(10.dp)
            .border(1.dp, MaterialTheme.colorScheme.primary, MaterialTheme.shapes.medium)
            .padding(10.dp)
            .fillMaxWidth()
    ) {
        LabelValue("Connected", status.isConnected.toString())
        LabelValue("Motor Status", status.motorStatus.label)
        LabelValue("Actual Motor Status", status.actualMotorStatus.label)
        LabelValue("ccValue", "${status.ccValue}")
        LabelValue("Position", "${status.positionMm} mm")
    }
}

@Composable
fun Goto(status: Status) {
    var interacting by remember { mutableStateOf(false) }
    var target by remember { mutableLongStateOf(-1L) }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 10.dp, end = 10.dp, bottom = 10.dp)
    ) {
        Row(
            modifier = Modifier
                .fillMaxWidth()
                .padding(16.dp),
        ) {
            Slider(
                modifier = Modifier.fillMaxWidth(),
                valueRange = 0f..ARM_LENGTH.toFloat(),
                value = if ( interacting || target > 0 )
                        target.toFloat()
                    else
                        status.positionMm.toFloat(),
                onValueChange = {
                    interacting = true
                    target = it.toLong()
                },
                onValueChangeFinished = {
                    interacting = false
                    Log.d(TAG, "Goto requested $target mm.")
                    app.controller.goto(target) {
                        target = -1L
                        Log.d(TAG, "Goto request completed or aborted $it mm.")
                    }
                }
            )
        }
        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.Center,
        ) {
            if (target >= 0) {
                Text(
                    text = "Moving to target position $target mm"
                )
            } else {
                Text(
                    text = "Current position ${status.positionMm} mm"
                )
            }
        }
    }
}

@Composable
fun NavigationScreen(viewModel: NavigationScreenViewModel = viewModel<NavigationScreenViewModel>()) {
    val appViewModel = LocalApplicationViewModel.current
    val tag = viewModel.classifierPreviewModel.tagFlow.collectAsState()
    val status by appViewModel.status.collectAsState()
    val state by appViewModel.applicationState.collectAsState()
    val prefs by appViewModel.prefs.collectAsState()

    appViewModel.updateApplicationState {
        it.copy(hideBottomBar = false)
    }

    DisposableEffect(LocalContext.current) {
        Log.d(TAG, "Initializing controller.")
        app.controller.statusCallback = appViewModel::updateStatus

        onDispose {
            Log.d(TAG, "Disconnecting controller.")
            app.controller.disconnect()
            app.controller.statusCallback = null
        }
    }

    Column(
        modifier = Modifier.fillMaxSize(),
    ) {
        Box(
            modifier = Modifier
                .fillMaxWidth()
                .aspectRatio(Classifier.ASPECT_RATIO)
                .clipToBounds()
        ) {
            // The preview is at zIndex 1f, the labels at 2f and the capture button at 3f.
            ClassifierPreview(viewModel.classifierPreviewModel)
            if (state.classifierEnabled && tag.value != null) {
                Image(
                    modifier = Modifier
                        .zIndex(2f)
                        .fillMaxSize()
                        .alpha(0.5f),
                    bitmap = tag.value!!.second.asImageBitmap(),
                    contentDescription = null,
                )
            }
            IconButton(
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .zIndex(3f)
                    .padding(8.dp)
                    .border(1.dp, MaterialTheme.colorScheme.onPrimary, MaterialTheme.shapes.small)
                    .background(color = MaterialTheme.colorScheme.primary, shape = MaterialTheme.shapes.small),
                onClick= { viewModel.captureCount = prefs.captureCount }
            ) {
                if ( viewModel.captureCount != 0) {
                    Text(
                        "${viewModel.captureCount}",
                        color = MaterialTheme.colorScheme.onPrimary
                    )
                } else {
                    Icon(
                        painter = painterResource(id = R.drawable.ic_camera),
                        contentDescription = "Capture image and labels.",
                        tint=MaterialTheme.colorScheme.onPrimary
                    )
                }
            }

        }
        LabelValue(
            Modifier
                .padding(10.dp)
                .fillMaxWidth(),
            "Median",
            if (tag.value == null) "" else "%1.2f".format(tag.value!!.first),
        )
        Controls()
        Spacer(Modifier.weight(1f))
        if (status.isConnected) {
            Status(status)
            Goto(status)
        }
    }
}
