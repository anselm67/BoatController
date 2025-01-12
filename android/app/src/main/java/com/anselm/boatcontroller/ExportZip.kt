package com.anselm.boatcontroller

import android.net.Uri
import android.util.Log
import com.anselm.boatcontroller.BoatControllerApplication.Companion.app
import java.io.File
import java.io.FileInputStream
import java.io.FileOutputStream
import java.time.LocalDate
import java.time.format.DateTimeFormatter
import java.util.zip.ZipEntry
import java.util.zip.ZipOutputStream

private fun exportZipFile(zipFile: File, dest: Uri, progress: ((Float) -> Unit)? = null) {
    val zipOut = ZipOutputStream(zipFile.outputStream())
    val files = app.listCaptureFiles()

    var done = 0
    zipOut.use { zip ->
        files.forEach { file ->
            zip.putNextEntry(ZipEntry(file.path))
            FileInputStream(file).use { inputStream -> inputStream.copyTo(zip) }
            progress?.let { it(done++.toFloat() / files.size) }
        }
    }
    // Copies the temp file into the provided destination.
    app.contentResolver.openFileDescriptor(dest, "w").use { fd ->
        FileOutputStream(fd?.fileDescriptor).use {
            FileInputStream(zipFile).copyTo(it)
        }
    }
}

fun exportFiles(dest: Uri, progress: ((Float) -> Unit)? = null) {
    val zipFile = File(app.cacheDir, "boat-controller.zip")
    try {
        return exportZipFile(zipFile, dest, progress)
    } catch (e: Exception) {
        Log.d("SettingsScreen.exportFiles", "Failed to export rides as zip file.", e)
    } finally {
        zipFile.delete()
    }
}

private val YYYYMMDDFormatter = DateTimeFormatter.ofPattern("yyyy-MM-dd")

fun makeDatedName(name: String): String {
    val now = LocalDate.now()
    return "${now.format(YYYYMMDDFormatter)}-$name"
}
