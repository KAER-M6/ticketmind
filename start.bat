@echo off
chcp 65001 >nul
title TicketMind 工单智脑 - 启动中
cd /d "%~dp0"
setlocal
set "PORT=8000"
set "APPMOD=app.main:app"
set "PYTHONPATH=%~dp0"
set "CURL=%SystemRoot%\System32\curl.exe"

echo ============================================
echo   工单智脑 TicketMind
echo ============================================
echo.

REM ---------- 1) 已在运行则只开浏览器 ----------
netstat -ano | findstr /c:":%PORT% " | findstr /c:"LISTENING" >nul
if not errorlevel 1 goto already_running

REM ---------- 2) 环境自检 ----------
if not exist "%~dp0app\main.py" goto err_app

set "PY="
if exist "%~dp0.venv\Scripts\python.exe" set "PY=%~dp0.venv\Scripts\python.exe"
if not defined PY if exist "%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe" set "PY=%USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
if not defined PY goto err_python
if not exist "%PY%" goto err_python

"%PY%" -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 goto err_deps

echo   Python   : %PY%
echo   端口     : %PORT%
echo.

REM ---------- 3) 启动服务（独立最小化窗口）----------
echo   [1/2] 正在启动服务，首次启动需加载检索索引，请稍候...
start "TicketMind Server" /min "%PY%" -m uvicorn %APPMOD% --host 127.0.0.1 --port %PORT%

REM ---------- 4) 轮询就绪（标签置于顶层，勿放进括号块）----------
set /a n=0
:wait
set /a n+=1
ping -n 2 127.0.0.1 >nul
"%CURL%" --noproxy "*" -s -m 2 -o nul "http://127.0.0.1:%PORT%/api/health" >nul 2>&1
if not errorlevel 1 goto ready
if %n% lss 60 goto wait
goto err_timeout

:ready
echo   [2/2] 服务已就绪（用时 %n% 秒），正在打开浏览器...
start "" "http://127.0.0.1:%PORT%/"
echo.
echo ============================================
echo   启动成功，浏览器已打开 http://127.0.0.1:%PORT%/
echo.
echo   停止服务：关闭任务栏中「TicketMind Server」窗口
echo             或双击本目录下的 stop.bat
echo ============================================
ping -n 5 127.0.0.1 >nul
exit /b 0

:already_running
echo   服务已在运行，无需重复启动。
start "" "http://127.0.0.1:%PORT%/"
echo   已为你打开浏览器：http://127.0.0.1:%PORT%/
echo   如需重启，请先双击 stop.bat 停止服务。
ping -n 5 127.0.0.1 >nul
exit /b 0

:err_app
echo   [错误] 未找到 app\main.py
echo   请确认本脚本与项目文件在同一目录（当前位置：%~dp0）
pause
exit /b 1

:err_python
echo   [错误] 未找到可用的 Python 解释器
echo   已尝试路径：
echo     %~dp0.venv\Scripts\python.exe
echo     %USERPROFILE%\.workbuddy\binaries\python\envs\default\Scripts\python.exe
pause
exit /b 1

:err_deps
echo   [错误] Python 缺少运行依赖：%PY%
echo   请执行安装： "%PY%" -m pip install -r requirements.txt
pause
exit /b 1

:err_timeout
echo   [错误] 等待 %n% 秒后服务仍未就绪。
echo   请点开任务栏中「TicketMind Server」窗口查看具体报错（常见原因：
echo   未配置 .env 中的 DEEPSEEK_API_KEY，或索引未构建）。
pause
exit /b 1
