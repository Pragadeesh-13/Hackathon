@echo off
REM Cattle Breed API Server - Windows Firewall Configuration
REM Run this script as Administrator

echo ========================================
echo Cattle Breed API Server Setup
echo ========================================
echo.
echo Configuring Windows Firewall for API Server...
echo.

REM Add inbound rule for port 5000
netsh advfirewall firewall add rule name="Cattle Breed API Server - Inbound" dir=in action=allow protocol=TCP localport=5000

REM Add outbound rule for port 5000 (if needed)
netsh advfirewall firewall add rule name="Cattle Breed API Server - Outbound" dir=out action=allow protocol=TCP localport=5000

echo.
echo ========================================
echo Firewall Configuration Complete!
echo ========================================
echo.
echo Your API server will be accessible at:
echo   Local: http://localhost:5000
echo   Network: http://172.16.45.105:5000
echo.
echo For Mac React app, use: http://172.16.45.105:5000
echo.
echo To start the API server, run:
echo   python api_server.py
echo.
pause