@echo off
chcp 65001 >nul
title Trikotillomani Kontrol - Kurulum
cd /d "%~dp0"
echo ==============================================
echo   Trikotillomani Kontrol - Kurulum
echo ==============================================
where py >nul 2>nul
if %errorlevel%==0 (set PY=py -3) else (set PY=python)
%PY% --version >nul 2>nul
if errorlevel 1 (
  echo.
  echo [HATA] Python bulunamadi.
  echo https://www.python.org/downloads/ adresinden Python 3.12 kurun.
  echo Kurulumda "Add python.exe to PATH" kutusunu isaretleyin, sonra bu dosyayi tekrar calistirin.
  pause
  exit /b 1
)
echo Sanal ortam olusturuluyor...
%PY% -m venv venv || (echo [HATA] venv olusturulamadi & pause & exit /b 1)
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt || (echo [HATA] Paketler kurulamadi & pause & exit /b 1)
echo.
echo Kurulum tamamlandi. Programi baslatmak icin baslat.bat dosyasini calistirin.
pause
