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

# Check if we're running on Linux
if [ "$(uname)" != "Linux" ]; then
    echo "❌ Error: This project only supports Linux compilation"
    echo "   Please run this script on a Linux system"
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

# Check for C++ compiler
if ! command -v g++ &> /dev/null && ! command -v clang++ &> /dev/null; then
    echo "❌ C++ compiler not found. Please install a C++ compiler:"
    echo "   Ubuntu/Debian: sudo apt-get install build-essential"
    exit 1
fi

# Check for CMake
if ! command -v cmake &> /dev/null; then
    echo "❌ CMake not found. Please install CMake:"
    echo "   Ubuntu/Debian: sudo apt-get install cmake"
    exit 1
fi

# Check for SDL2
if ! pkg-config --exists sdl2; then
    echo "❌ SDL2 not found. Please install SDL2:"
    echo "   Ubuntu/Debian: sudo apt-get install libsdl2-dev"
    exit 1
fi

# Check for ALSA
if ! pkg-config --exists alsa; then
    echo "❌ ALSA not found. Please install ALSA development libraries:"
    echo "   Ubuntu/Debian: sudo apt-get install libasound2-dev"
    exit 1
fi

# Check for nlohmann/json
if ! pkg-config --exists nlohmann_json; then
    # Try to find it manually
    if [ ! -f "/usr/include/nlohmann/json.hpp" ] && [ ! -f "/usr/local/include/nlohmann/json.hpp" ]; then
        echo "❌ nlohmann/json not found. Please install nlohmann/json:"
        echo "   Ubuntu/Debian: sudo apt-get install nlohmann-json3-dev"
        echo "   Or download header-only version from: https://github.com/nlohmann/json"
        exit 1
    fi
fi

# Check for git (needed for submodules)
if ! command -v git &> /dev/null; then
    echo "❌ Git not found. Please install Git:"
    echo "   Ubuntu/Debian: sudo apt-get install git"
    exit 1
fi

# Check for make
if ! command -v make &> /dev/null; then
    echo "❌ Make not found. Please install Make:"
    echo "   Ubuntu/Debian: sudo apt-get install build-essential"
    exit 1
fi

# Check for tar (needed for dependency extraction)
if ! command -v tar &> /dev/null; then
    echo "❌ Tar not found. Please install tar:"
    echo "   Ubuntu/Debian: sudo apt-get install tar"
    exit 1
fi


echo "✅ System dependencies OK"

# Initialize git submodules
echo "📦 Initializing git submodules..."
git submodule update --init
echo "   - Paroli TTS submodule initialized"

# Build the project
echo "🔨 Building the project..."
make clean
make build

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