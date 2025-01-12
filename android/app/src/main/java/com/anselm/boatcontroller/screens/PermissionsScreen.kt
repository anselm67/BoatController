package com.anselm.boatcontroller.screens

import android.content.Intent
import android.net.Uri
import android.provider.Settings
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.border
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.paint
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import com.anselm.boatcontroller.BoatControllerApplication.Companion.app
import com.anselm.boatcontroller.LocalNavController
import com.anselm.boatcontroller.NavigationItem
import com.anselm.boatcontroller.R
import com.anselm.boatcontroller.components.AppButton

@Composable
fun PermissionsScreen() {
    val navController = LocalNavController.current
    val context = LocalContext.current

    val launcher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()) { it ->
        if ( it.all { it.value } ) {
            navController.navigate(NavigationItem.Navigation.route)
        }
    }

    DisposableEffect(LocalContext.current) {
        if ( ! app.checkPermissions() )
            launcher.launch(app.allPermissions)

        onDispose {  }
    }

    Column (
        modifier = Modifier
            .fillMaxSize()
            .paint(
                painter = painterResource(id = R.drawable.permission_background),
                contentScale = ContentScale.FillHeight,
                alpha = 0.45f
            )
            .padding(16.dp),
        verticalArrangement = Arrangement.Center,
    ) {
        Column (
            modifier = Modifier
                .border(
                    width = 2.dp,
                    color = MaterialTheme.colorScheme.primary,
                    shape = RoundedCornerShape(10.dp)
                )
                .padding(16.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
        ) {
            Spacer(modifier = Modifier.height(32.dp))
            Text(
                text = "Permissions required",
                style = MaterialTheme.typography.displaySmall,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold
            )
            Spacer(modifier = Modifier.height(24.dp))
            Text(
                text = "In order to use this application, you must grant it location permission" +
                        " while using the app.",
                style = MaterialTheme.typography.bodyLarge,
                color = MaterialTheme.colorScheme.primary,
                fontWeight = FontWeight.Bold,
            )
            AppButton(
                modifier = Modifier
                    .padding(top = 16.dp)
                    .align(Alignment.CenterHorizontally),
                onClick = {
                    val intent = Intent(
                        Settings.ACTION_APPLICATION_DETAILS_SETTINGS,
                    )
                    intent.data = Uri.fromParts("package", context.packageName, null)
                    context.startActivity(intent)
                },
            ) {
                Text("Grant Permissions")
            }
            Spacer(modifier = Modifier.height(32.dp))
        }
    }
}
