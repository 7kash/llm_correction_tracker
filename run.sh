#!/bin/bash

# LLM Learning Visualizer - Quick Start Script

echo "🤖 LLM Learning Visualizer - Starting..."
echo ""

# Check if we're in the right directory
if [ ! -f "backend/app.py" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Error: Python 3 is not installed"
    echo "Please install Python 3.8 or higher from python.org"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "backend/venv" ]; then
    echo "📦 Creating virtual environment..."
    cd backend
    python3 -m venv venv
    cd ..
fi

# Activate virtual environment and install dependencies
echo "📦 Installing dependencies..."
cd backend

# Activate venv based on OS
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "win32" ]]; then
    source venv/Scripts/activate
else
    source venv/bin/activate
fi

pip install -q -r requirements.txt

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo ""
    echo "⚠️  No .env file found!"
    echo "📝 Creating .env from .env.example..."
    cp .env.example .env
    echo ""
    echo "⚠️  IMPORTANT: Please edit backend/.env and add your OpenAI API key!"
    echo "   Get your API key from: https://platform.openai.com/api-keys"
    echo ""
    echo "Press Enter after you've added your API key to continue..."
    read
fi

# Start the backend
echo "🚀 Starting Flask backend on http://localhost:5000..."
echo ""
python app.py &
BACKEND_PID=$!

# Wait a bit for backend to start
sleep 2

# Open the frontend
echo ""
echo "🌐 Opening frontend..."
echo ""

cd ../frontend

# Try to open the browser
if command -v xdg-open &> /dev/null; then
    xdg-open "http://localhost:8000" 2>/dev/null
elif command -v open &> /dev/null; then
    open "http://localhost:8000" 2>/dev/null
fi

# Start a simple HTTP server for the frontend
echo "🌐 Starting frontend server on http://localhost:8000..."
python3 -m http.server 8000 &
FRONTEND_PID=$!

echo ""
echo "✅ Application is running!"
echo ""
echo "   📱 Frontend: http://localhost:8000"
echo "   🔧 Backend:  http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop the servers..."
echo ""

# Wait for user to stop
trap "echo ''; echo '🛑 Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
