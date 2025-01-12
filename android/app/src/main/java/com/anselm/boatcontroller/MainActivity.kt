package com.anselm.boatcontroller

import android.os.Bundle
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.viewmodel.compose.viewModel
import androidx.navigation.NavHostController
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.anselm.boatcontroller.BoatControllerApplication.Companion.app
import com.anselm.boatcontroller.components.AppBottomBar
import com.anselm.boatcontroller.components.AppTopBar
import com.anselm.boatcontroller.models.ApplicationViewModel
import com.anselm.boatcontroller.models.LocalApplicationViewModel
import com.anselm.boatcontroller.screens.NavigationScreen
import com.anselm.boatcontroller.screens.PermissionsScreen
import com.anselm.boatcontroller.screens.SettingsScreen
import com.anselm.boatcontroller.ui.theme.AppTheme

val LocalNavController = compositionLocalOf<NavHostController> {
    error("No NavController found.")
}

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AppTheme(dynamicColor = false) {
                val navController = rememberNavController()
                val appViewModel: ApplicationViewModel =
                    viewModel(factory = ApplicationViewModel.Factory())
                CompositionLocalProvider(LocalApplicationViewModel provides appViewModel) {
                    CompositionLocalProvider(LocalNavController provides navController) {
                        KeepScreenOn()
                        MainScreen()
                    }
                }
            }
        }
    }

    @Composable
    private fun KeepScreenOn() {
        val appViewModel = LocalApplicationViewModel.current
        val prefs by appViewModel.prefs.collectAsState()

        LaunchedEffect(prefs) {
            if (prefs.alwaysOn) {
                window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            } else {
                window.clearFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
            }
        }
    }

    @Composable
    private fun MainScreen() {
        val appViewModel = LocalApplicationViewModel.current
        val navController = LocalNavController.current

        Scaffold(
            topBar = { AppTopBar(appViewModel) },
            bottomBar = { AppBottomBar(appViewModel) }
        ) { innerPadding ->
            NavHost(
                modifier = Modifier.padding(innerPadding),
                navController = navController,
                startDestination = if ( app.checkPermissions() )
                    NavigationItem.Navigation.route
                else
                    NavigationItem.Permissions.route
            ) {
                composable(NavigationItem.Permissions.route) {
                    PermissionsScreen()
                }
                composable(NavigationItem.Navigation.route) {
                    NavigationScreen()
                }
                composable(NavigationItem.Settings.route) {
                    SettingsScreen()
                }
            }
        }
    }
}
