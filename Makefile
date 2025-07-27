# Local LLM Makefile
# Provides convenient shortcuts for common development tasks

.PHONY: help build clean setup run test

# Default target
help:
	@echo "Local LLM - Available targets:"
	@echo "  build    - Build the project"
	@echo "  clean    - Clean build artifacts"
	@echo "  setup    - Run setup script"
	@echo "  run      - Build and run the application"
	@echo "  test     - Run tests (if available)"

# Build the project
build:
	@echo "🔨 Building project..."
	mkdir -p build
	cd build && cmake .. && make -j$(shell nproc)

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	rm -rf build/
	rm -f *.o *.so *.dylib *.dll *.exe

# Run setup script
setup:
	@echo "🚀 Running setup..."
	chmod +x scripts/setup.sh
	./scripts/setup.sh

# Build and run
run: build
	@echo "🎯 Running application..."
	./build/local-llm

# Install dependencies (Ubuntu/Debian)
install-deps-ubuntu:
	@echo "📦 Installing Ubuntu dependencies..."
	sudo apt-get update
	sudo apt-get install -y cmake libsdl2-dev python3 python3-pip python3-venv alsa-utils

# Install dependencies (macOS)
install-deps-macos:
	@echo "📦 Installing macOS dependencies..."
	brew install cmake sdl2 python3

# Create model directories
create-dirs:
	@echo "📁 Creating model directories..."
	mkdir -p models/stt models/llm models/tts sessions

# Format code (requires clang-format)
format:
	@echo "🎨 Formatting code..."
	find src/ include/ -name "*.cpp" -o -name "*.h" | xargs clang-format -i

# Check code style (requires clang-format)
check-format:
	@echo "🔍 Checking code style..."
	find src/ include/ -name "*.cpp" -o -name "*.h" | xargs clang-format --dry-run --Werror 