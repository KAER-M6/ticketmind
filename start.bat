@echo off
title TicketMind 工单智脑
cd /d %~dp0
set PYTHONPATH=.

echo ============================================
echo   TicketMind 工单智脑 - 正在启动...
echo   服务窗口已单独打开，浏览器即将自动弹出
echo   关闭服务窗口即停止程序
echo ============================================

start "TicketMind Server" "C:\Users\王俊豪\.workbuddy\binaries\python\envs\default\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000

timeout /t 3 /nobreak >nul
start "" "http://127.0.0.1:8000"
exit
