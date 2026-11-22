Set WshShell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
strPath = fso.GetParentFolderName(WScript.ScriptFullName)

' Run agriGPT.bat hidden (0 = hide window, False = do not wait)
WshShell.Run """" & strPath & "\agriGPT.bat""", 0, False
