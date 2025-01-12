package com.anselm.boatcontroller

enum class Screen {
    PERMISSIONS,
    NAVIGATION,
    SETTINGS,
}

sealed class NavigationItem(val route: String) {
    data object Permissions: NavigationItem(Screen.PERMISSIONS.name)
    data object Navigation : NavigationItem(Screen.NAVIGATION.name)
    data object Settings : NavigationItem(Screen.SETTINGS.name)
}

