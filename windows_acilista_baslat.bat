@echo off
chcp 65001 >nul
rem Bilgisayar acildiginda programin otomatik baslamasi icin Baslangic klasorune kisayol ekler.
cd /d "%~dp0"
set "HEDEF=%~dp0TrikoKontrol.exe"
if not exist "%HEDEF%" set "HEDEF=%~dp0baslat.bat"
set "KISAYOL=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\TrikoKontrol.lnk"
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut('%KISAYOL%'); $s.TargetPath='%HEDEF%'; $s.WorkingDirectory='%~dp0'; $s.Save()"
if exist "%KISAYOL%" (echo Tamam: Program artik Windows acilisinda otomatik baslayacak.) else (echo Kisayol olusturulamadi.)
echo Kaldirmak icin su dosyayi silin: %KISAYOL%
pause
