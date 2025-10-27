@echo off
REM LLM Learning Visualizer - Quick Start Script for Windows

echo 🤖 LLM Learning Visualizer - Starting...
echo.

REM Check if we're in the right directory
if not exist "backend\app.py" (
    echo ❌ Error: Please run this script from the project root directory
    pause
    exit /b 1
)

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Error: Python is not installed
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)

REM Check if virtual environment exists
if not exist "backend\venv" (
    echo 📦 Creating virtual environment...
    cd backend
    python -m venv venv
    cd ..
)

REM Activate virtual environment and install dependencies
echo 📦 Installing dependencies...
cd backend
call venv\Scripts\activate.bat
pip install -q -r requirements.txt

REM Check if .env file exists
if not exist ".env" (
    echo.
    echo ⚠️  No .env file found!
    echo 📝 Creating .env from .env.example...
    copy .env.example .env
    echo.
    echo ⚠️  IMPORTANT: Please edit backend\.env and add your OpenAI API key!
    echo    Get your API key from: https://platform.openai.com/api-keys
    echo.
    pause
)

REM Start the backend
echo 🚀 Starting Flask backend on http://localhost:5000...
echo.
start /B python app.py

REM Wait for backend to start
timeout /t 2 /nobreak >nul

REM Open the frontend
echo.
echo 🌐 Opening frontend...
cd ..\frontend

REM Start HTTP server for frontend
echo 🌐 Starting frontend server on http://localhost:8000...
start /B python -m http.server 8000

REM Open in browser
timeout /t 2 /nobreak >nul
start http://localhost:8000

echo.
echo ✅ Application is running!
echo.
echo    📱 Frontend: http://localhost:8000
echo    🔧 Backend:  http://localhost:5000
echo.
echo Press any key to stop the servers...
pause >nul

REM Kill Python processes (backend and frontend servers)
taskkill /F /IM python.exe /T >nul 2>&1

echo.
echo 🛑 Servers stopped.
pause
