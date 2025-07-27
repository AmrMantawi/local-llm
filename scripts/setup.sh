#!/bin/bash

# Local LLM Setup Script
# This script helps set up the development environment for the Local LLM project

set -e

echo "🚀 Setting up Local LLM development environment..."

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p build
mkdir -p models/llm
mkdir -p models/tts
mkdir -p models/stt
mkdir -p sessions

# Check for required system dependencies
echo "🔍 Checking system dependencies..."

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "❌ CMake not found. Please install CMake:"
    echo "   Ubuntu/Debian: sudo apt-get install cmake"
    echo "   macOS: brew install cmake"
    exit 1
fi

# Check for SDL2
if ! pkg-config --exists sdl2; then
    echo "❌ SDL2 not found. Please install SDL2:"
    echo "   Ubuntu/Debian: sudo apt-get install libsdl2-dev"
    echo "   macOS: brew install sdl2"
    exit 1
fi

# Check for Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.7+"
    exit 1
fi

# Check for pip
if ! command -v pip3 &> /dev/null; then
    echo "❌ pip3 not found. Please install pip"
    exit 1
fi

echo "✅ System dependencies OK"

# Initialize git submodules
echo "📦 Initializing git submodules..."
git submodule update --init --recursive

# Create virtual environment
echo "🐍 Setting up Python virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install --upgrade pip
pip install piper-tts

# Make scripts executable
echo "🔧 Making scripts executable..."
chmod +x scripts/speak

# Build the project
echo "🔨 Building the project..."
cd build
cmake ..
make -j$(nproc)

echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Download models to the models/ directory:"
echo "   - LLM: Place .gguf files in models/llm/"
echo "   - TTS: Place .onnx files in models/tts/"
echo "   - STT: Place .bin files in models/stt/"
echo ""
echo "2. Run the application:"
echo "   ./build/local-llm"
echo ""
echo "3. For help with models, see the README.md file" 