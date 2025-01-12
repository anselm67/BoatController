package com.anselm.boatcontroller.components

import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.RowScope
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonColors
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.ButtonElevation
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Shape
import androidx.compose.ui.unit.dp

@Composable
fun AppButton(
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    colors: ButtonColors = ButtonDefaults.buttonColors(),
    elevation: ButtonElevation = ButtonDefaults.buttonElevation(),
    contentPadding: PaddingValues = PaddingValues(horizontal = 16.dp),
    enabled: Boolean = true,
    interactionSource: MutableInteractionSource = remember { MutableInteractionSource() },
    shape: Shape = MaterialTheme.shapes.small, // Explicitly pass shape here
    content: @Composable RowScope.() -> Unit
) {
    // Pass everything except 'shape' down to the Button
    Button(
        onClick = onClick,
        modifier = modifier,
        colors = colors,
        elevation = elevation,
        contentPadding = contentPadding,
        enabled = enabled,
        interactionSource = interactionSource,
        shape = shape,
        content = content
    )
}
