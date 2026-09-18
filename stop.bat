@echo off
chcp 65001 >nul
title TicketMind 工单智脑 - 停止服务
setlocal
set "PORT=8000"

echo ============================================
echo   工单智脑 TicketMind - 停止服务
echo ============================================
echo.

netstat -ano | findstr /c:":%PORT% " | findstr /c:"LISTENING" >nul
if errorlevel 1 goto not_running

echo   正在停止占用端口 %PORT% 的进程...
for /f "tokens=5" %%a in ('netstat -ano ^| findstr /c:":%PORT% " ^| findstr /c:"LISTENING"') do (
    taskkill /f /pid %%a >nul 2>&1
    if not errorlevel 1 echo     已终止进程 PID=%%a
)
ping -n 3 127.0.0.1 >nul

netstat -ano | findstr /c:":%PORT% " | findstr /c:"LISTENING" >nul
if errorlevel 1 goto stopped

echo   [警告] 端口 %PORT% 仍被占用，可能由其他程序持有：
netstat -ano | findstr /c:":%PORT% " | findstr /c:"LISTENING"
pause
exit /b 1

:stopped
echo.
echo   服务已停止，端口 %PORT% 已释放。
ping -n 4 127.0.0.1 >nul
exit /b 0

:not_running
echo   服务未在运行（端口 %PORT% 空闲）。
ping -n 4 127.0.0.1 >nul
exit /b 0
