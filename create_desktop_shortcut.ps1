# Creates a Windows Desktop shortcut named "agriGPT" with custom icon
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$desktopPath = [Environment]::GetFolderPath("Desktop")
$shortcutPath = Join-Path -Path $desktopPath -ChildPath "agriGPT.lnk"

$WshShell = New-Object -ComObject WScript.Shell
$Shortcut = $WshShell.CreateShortcut($shortcutPath)

# Point to silent VBS launcher or BAT launcher
$targetVbs = Join-Path -Path $scriptDir -ChildPath "agriGPT.vbs"
$targetBat = Join-Path -Path $scriptDir -ChildPath "agriGPT.bat"
$iconPath = Join-Path -Path $scriptDir -ChildPath "assets\agriGPT.ico"

if (Test-Path $targetVbs) {
    $Shortcut.TargetPath = "wscript.exe"
    $Shortcut.Arguments = "`"$targetVbs`""
} else {
    $Shortcut.TargetPath = $targetBat
}

$Shortcut.WorkingDirectory = $scriptDir
$Shortcut.Description = "agriGPT - AI Agricultural Assistant & Plant Pathology Lab"

if (Test-Path $iconPath) {
    $Shortcut.IconLocation = "$iconPath,0"
}

$Shortcut.Save()
Write-Host "SUCCESS: Created Desktop shortcut '$shortcutPath'"
