package com.anselm.boatcontroller.components

import android.util.Log
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.BottomAppBar
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import com.anselm.boatcontroller.BoatControllerApplication.Companion.app
import com.anselm.boatcontroller.TAG
import com.anselm.boatcontroller.controller.MotorStatus
import com.anselm.boatcontroller.controller.Status
import com.anselm.boatcontroller.models.ApplicationViewModel

@Composable
fun AppBottomBar(appViewModel: ApplicationViewModel) {
    val status by appViewModel.status.collectAsState(initial = Status.unknown)
    val state by appViewModel.applicationState.collectAsState()

    if ( state.hideBottomBar ) {
        return
    }

    BottomAppBar(
        containerColor = MaterialTheme.colorScheme.primaryContainer,
        contentColor = MaterialTheme.colorScheme.primary,
    ) {
        if (status.isConnected) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceAround,
            ) {
                AppButton(onClick = {
                    Log.d(TAG, "Left ...")
                    app.controller.left {
                        Log.d(TAG, "Finished left.")
                    }
                }, enabled = status.motorStatus != MotorStatus.Left ) {
                    Text(text = "Left ")
                }
                AppButton(onClick = {
                    Log.d(TAG, "Stop ...")
                    app.controller.stop()
                }, enabled = status.motorStatus != MotorStatus.Off ) {
                    Text(text = "Stop")
                }
                AppButton(onClick = {
                    Log.d(TAG, "Right ...")
                    app.controller.right {
                        Log.d(TAG, "Finished right..")
                    }
                }, enabled = status.motorStatus != MotorStatus.Right ) {
                    Text(text = "Right ")
                }
            }
        } else {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(10.dp),
                horizontalArrangement = Arrangement.Center,
            ) {
                AppButton(onClick = {
                    Log.d(TAG, "Connecting ...")
                    app.controller.connect()
                }) {
                    Text(text = "Connect")
                }
            }
        }
    }
}