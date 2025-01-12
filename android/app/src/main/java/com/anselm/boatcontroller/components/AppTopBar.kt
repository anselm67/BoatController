package com.anselm.boatcontroller.components

import androidx.compose.foundation.layout.size
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.filled.ArrowBack
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.TopAppBarDefaults
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.res.painterResource
import androidx.compose.ui.unit.dp
import androidx.navigation.compose.currentBackStackEntryAsState
import com.anselm.boatcontroller.LocalNavController
import com.anselm.boatcontroller.NavigationItem
import com.anselm.boatcontroller.R
import com.anselm.boatcontroller.models.ApplicationViewModel

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun AppTopBar(appViewModel: ApplicationViewModel) {
    val navController = LocalNavController.current
    val state by appViewModel.applicationState.collectAsState()

    TopAppBar(
        colors = TopAppBarDefaults.topAppBarColors(
            containerColor = MaterialTheme.colorScheme.primary,
            titleContentColor = MaterialTheme.colorScheme.onPrimary,
        ),
        title = {
            Text(
                text = state.title,
                maxLines = 1,
            )
        },
        actions = {
            IconButton(
                onClick = { navController.navigate(NavigationItem.Settings.route) },
                modifier = Modifier.size(32.dp),
            ) {
                Icon(
                    modifier = Modifier.size(32.dp),
                    painter= painterResource(id = R.drawable.ic_settings),
                    contentDescription = "Settings",
                    tint = MaterialTheme.colorScheme.onPrimary
                )
            }
        },
        navigationIcon = {
            val currentBackStackEntry by navController.currentBackStackEntryAsState()
            val canNavigateBack = currentBackStackEntry?.destination?.route != NavigationItem.Navigation.route
            if ( canNavigateBack ) {
                IconButton(onClick = { navController.navigateUp() }) {
                    Icon(
                        imageVector = Icons.AutoMirrored.Filled.ArrowBack,
                        contentDescription = "Back",
                        tint = MaterialTheme.colorScheme.onPrimary
                    )
                }
            }
        }
    )
}