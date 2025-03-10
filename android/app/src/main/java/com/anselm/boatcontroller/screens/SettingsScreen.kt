package com.anselm.boatcontroller.screens

import android.net.Uri
import android.util.Log
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.SegmentedButton
import androidx.compose.material3.SegmentedButtonDefaults
import androidx.compose.material3.SingleChoiceSegmentedButtonRow
import androidx.compose.material3.Slider
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.viewmodel.compose.viewModel
import com.anselm.boatcontroller.BoatControllerApplication.Companion.app
import com.anselm.boatcontroller.BuildConfig
import com.anselm.boatcontroller.R
import com.anselm.boatcontroller.exportFiles
import com.anselm.boatcontroller.makeDatedName
import com.anselm.boatcontroller.models.LocalApplicationViewModel
import com.anselm.boatcontroller.models.SettingsScreenViewModel
import kotlinx.coroutines.launch

@Composable
private fun DeleteCapturedFiles(capturedCount: Int, done: () -> Unit) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 8.dp, bottom = 8.dp, top = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text("Delete $capturedCount files?")
        IconButton(
            modifier = Modifier.border(
                2.dp,
                MaterialTheme.colorScheme.primary,
                shape = MaterialTheme.shapes.small),
            onClick = {
                app.deleteCapturedFiles()
                done()
        }) {
            Icon(
                painter = painterResource(id = R.drawable.ic_trash),
                contentDescription = "Run this action.",
                tint = MaterialTheme.colorScheme.primary,
                modifier = Modifier.size(24.dp)
            )
        }
    }
}


@Composable
private fun ExportCapturedFiles(viewModel: SettingsScreenViewModel, capturedCount: Int) {
    val launcher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.CreateDocument("application/zip"),
        onResult = { uri: Uri? ->
            if (uri != null) {
                viewModel.showProgress = true
                app.applicationScope.launch {
                    try {
                        exportFiles(uri)  { viewModel.progress = it }
                    } catch (e: Exception) {
                        Log.e("SettingsScreen.exportFiles", "Failed to export rides.", e)
                    }
                }.invokeOnCompletion {
                    // We're running on the application lifecycle scope, so this view that we're
                    viewModel.showProgress = false
                    app.toast("All images and labels exported.")
                }
            }
        }
    )

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(start = 8.dp, bottom = 8.dp, top = 8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text("Export $capturedCount files to zip?")
        if ( viewModel.showProgress ) {
            LinearProgressIndicator(
                modifier = Modifier.fillMaxWidth(),
                progress = { viewModel.progress }
            )
        } else {
            IconButton(
                modifier = Modifier.border(
                    2.dp,
                    MaterialTheme.colorScheme.primary,
                    shape = MaterialTheme.shapes.small
                ),
                onClick = { launcher.launch(makeDatedName("imgtags.zip")) },
            ) {
                Icon(
                    painter = painterResource(id = R.drawable.ic_zip),
                    contentDescription = "Run this action.",
                    tint = MaterialTheme.colorScheme.primary,
                    modifier = Modifier.size(24.dp)
                )
            }
        }
    }
}

@Composable
private fun EditCaptureCount(viewModel: SettingsScreenViewModel) {
    val appViewModel = LocalApplicationViewModel.current

    Row(
        modifier = Modifier
            .padding(vertical = 8.dp)
            .fillMaxWidth(),
        horizontalArrangement = Arrangement.Center
    ) {
        Text("Capture ${viewModel.captureCount} images")
    }
    Row(Modifier.fillMaxWidth()) {
        Slider(
            modifier = Modifier
                .padding(horizontal = 16.dp)
                .fillMaxWidth(),
            value = viewModel.captureCount.toFloat(),
            valueRange = 1f..10f,
            onValueChange = { viewModel.captureCount = it.toInt() },
            onValueChangeFinished = {
                appViewModel.updatePreferences { it.copy(
                    captureCount = viewModel.captureCount
                ) }
            }
        )
    }
}

@Composable
fun EditAlwaysOn(viewModel: SettingsScreenViewModel) {
    val appViewModel = LocalApplicationViewModel.current

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text = "Keep application on?")
        Switch(
            checked = viewModel.alwaysOn,
            onCheckedChange = {
                viewModel.alwaysOn = ! viewModel.alwaysOn
                appViewModel.updatePreferences { it.copy(
                    alwaysOn = viewModel.alwaysOn
                ) }
            }
        )
    }
}

@Composable
fun EditArmMount(viewModel: SettingsScreenViewModel) {
    val appViewModel = LocalApplicationViewModel.current

    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp, vertical=8.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text = "Arm mount")
        SingleChoiceSegmentedButtonRow {
            listOf("Left", "Right").forEachIndexed { index, label ->
                SegmentedButton(
                    shape = SegmentedButtonDefaults.itemShape(
                        index = index,
                        count = 2
                    ),
                    onClick = {
                        viewModel.armIndex = index
                        appViewModel.updatePreferences { it.copy(
                            leftMount = (index == 0)
                        ) }
                    },
                    selected = index == viewModel.armIndex,
                    label = { Text(label) }
                )
            }
        }
    }
}

@Composable
private fun CaptureSettings(viewModel: SettingsScreenViewModel) {
    var capturedCount by remember { mutableIntStateOf(app.captureFileCount) }

    Column(
        Modifier
            .padding(bottom = 8.dp)
            .fillMaxWidth()) {
        Row(
            Modifier
                .padding(vertical = 8.dp)
                .fillMaxWidth()) {
            Text("Capture Settings", fontWeight = FontWeight.Bold)
        }
        if ( capturedCount > 0 ) {
            DeleteCapturedFiles(capturedCount) { capturedCount = 0 }
            ExportCapturedFiles(viewModel, capturedCount)
        }
        EditCaptureCount(viewModel)
    }
}

@Composable
private fun ClassifierSettings(viewModel: SettingsScreenViewModel) {
    val appViewModel = LocalApplicationViewModel.current

    Row(
        Modifier
            .padding(vertical = 8.dp)
            .fillMaxWidth()) {
        Text("Classifier Settings [${BuildConfig.FLAVOR}]", fontWeight = FontWeight.Bold)
    }
    Row(
        modifier = Modifier
            .padding(vertical = 8.dp)
            .fillMaxWidth(),
        horizontalArrangement = Arrangement.Center
    ) {
        Text("Wait for ${viewModel.analysisDelayMillis} ms in between analysis")
    }
    Row(Modifier.fillMaxWidth()) {
        Slider(
            modifier = Modifier
                .padding(horizontal = 16.dp)
                .fillMaxWidth(),
            value = viewModel.analysisDelayMillis.toFloat(),
            valueRange = 250f..2000f,
            onValueChange = {
                val roundedMillis = (it / 50f).toLong() * 50L
                viewModel.analysisDelayMillis = roundedMillis
            },
            onValueChangeFinished = {
                appViewModel.updatePreferences { it.copy(
                    analysisDelayMillis = viewModel.analysisDelayMillis
                ) }
            }
        )
    }
}

@Composable
fun EditUseMockController(viewModel: SettingsScreenViewModel) {
    val appViewModel = LocalApplicationViewModel.current

    Row(
        Modifier
            .padding(vertical = 8.dp)
            .fillMaxWidth()) {
        Text("Debug Options", fontWeight = FontWeight.Bold)
    }
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .padding(horizontal = 16.dp),
        horizontalArrangement = Arrangement.SpaceBetween,
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(text = "Use mock controller (debug)?")
        Switch(
            checked = viewModel.useMockController,
            onCheckedChange = {
                viewModel.useMockController = ! viewModel.useMockController
                appViewModel.updatePreferences { it.copy(
                    useMockController = viewModel.useMockController
                ) }
            }
        )
    }
}

@Composable
fun SettingsScreen(viewModel: SettingsScreenViewModel = viewModel()) {
    val appViewModel = LocalApplicationViewModel.current
    val state by appViewModel.applicationState.collectAsState()

    appViewModel.updateApplicationState {
        state.copy(hideBottomBar = true)
    }

    Column(
        modifier = Modifier
            .padding(16.dp)
            .fillMaxSize()
    ) {
        EditAlwaysOn(viewModel)
        EditArmMount(viewModel)
        HorizontalDivider()
        CaptureSettings(viewModel)
        HorizontalDivider()
        ClassifierSettings(viewModel)
        HorizontalDivider()
        EditUseMockController(viewModel)
    }
}