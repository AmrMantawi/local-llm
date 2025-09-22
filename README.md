# Local LLM

A complete local AI assistant with Speech-to-Text (STT), Large Language Model (LLM), and Text-to-Speech (TTS) capabilities. Features a high-performance multi-threaded async pipeline architecture for real-time voice interaction.

## 🚀 Features

- **Real-time Voice Assistant**: Full Audio → STT → LLM → TTS pipeline
- **Multi-threaded Architecture**: Parallel processing with thread-safe queues
- **Multiple Pipeline Modes**: Voice assistant, text-only, transcription, synthesis
- **Speech-to-Text**: Whisper-based transcription with Voice Activity Detection (VAD)
- **Language Model**: Llama-based text generation with GGUF model support
- **Text-to-Speech**: Paroli TTS with ONNX neural voice synthesis
- **Server Mode**: Unix socket server for integration with other applications
- **Cross-platform**: Linux, Windows (WSL)
- **Privacy-first**: Everything runs locally, no cloud dependencies
- **Performance Optimized**: Release builds with -O3 optimizations and native CPU targeting

## 📋 Requirements

### System Dependencies
- **CMake** 3.16+
- **SDL2** development libraries
- **Python** 3.7+
- **ALSA** (Linux)
- **ONNX Runtime** (automatically downloaded)
- **nlohmann/json** (automatically downloaded)

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install cmake libsdl2-dev alsa-utils
```

## 🛠️ Quick Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/amrmantawi/local-llm.git
   cd local-llm
   ```

2. **Run the setup script:**
   ```bash
   make setup
   ```

3. **Download models:**
   - **STT**: Download Whisper models from [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp)
   - **LLM**: Download GGUF models from [Hugging Face](https://huggingface.co/models?search=gguf)
   - **TTS**: Download Paroli models (encoder.onnx, decoder.onnx, config.json) from [Paroli releases](https://github.com/paroli-ai/paroli-daemon/releases)

4. **Place models in the correct directories:**
   ```
   models/
   ├── stt/     # .bin files (Whisper)
   ├── llm/     # .gguf files (Llama)
   └── tts/     # .onnx files (Paroli) + config.json + espeak-ng-data/
   ```

5. **Build and run:**
   ```bash
   # Build optimized release version
   make build
   
   # Run in CLI mode (voice assistant)
   make run
   
   # Or run server mode
   ./build/local-llm --server --socket /run/local-llm.sock
   ```

## 🎯 Usage

### CLI Mode (Voice Assistant)
Interactive voice chat with real-time processing:
1. **Start the application** - initializes all components in parallel threads
2. **Speak into your microphone** - VAD automatically detects speech
3. **Wait for transcription** - Whisper converts speech to text
4. **Get AI response** - Llama generates contextual response
5. **Hear the response** - Paroli synthesizes and plays audio

### Server Mode
Unix socket server for integration with other applications:
```bash
./build/local-llm --server --socket /run/local-llm.sock
```

### Command Line Options
```bash
./build/local-llm [options]

Options:
  --server              Run in server mode (default: CLI mode)
  --config PATH         Path to models.json config file
  --socket PATH         Unix socket path for server mode
  --help, -h            Show help message
```

**Controls:**
- Press `Ctrl+C` to exit gracefully (proper cleanup of all threads and audio devices)

## 🏗️ Architecture

### Multi-threaded Pipeline
The system uses a sophisticated async pipeline architecture:

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  STTProcessor   │───▶│  LLMProcessor   │───▶│  TTSProcessor   │───▶│AudioOutputProc │
│  (Audio→Text)   │    │  (Text→Text)    │    │  (Text→Audio)   │    │  (Audio Play)  │
│  SDL2 + Whisper │    │  Llama.cpp      │    │  Paroli TTS     │    │  ALSA Output   │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Pipeline Modes
- **VOICE_ASSISTANT**: Full Audio → STT → LLM → TTS chain
- **TEXT_ONLY**: LLM only for server mode
- **TRANSCRIPTION**: Audio → STT → Text
- **SYNTHESIS**: Text → TTS → Audio

### Performance Optimizations
- **Thread-safe Queues**: Bounded queues with interrupt support
- **Signal-based Control**: Immediate interruption and graceful shutdown
- **Memory Management**: RAII with smart pointers, move semantics
- **Audio Optimization**: Low-latency ALSA with immediate interruption
- **GPU Support**: Llama supports GPU layer offloading
- **Release Optimizations**: -O3, -march=native, -DNDEBUG by default

## 📁 Project Structure

```
local-llm/
├── include/                    # Header files
│   ├── async_pipeline.h        # Core async pipeline framework
│   ├── async_processors.h      # Processor implementations
│   ├── pipeline_manager.h      # Pipeline coordination
│   ├── async_pipeline_factory.h # Factory for pipeline creation
│   ├── stt.h                   # STT interface
│   ├── llm.h                   # LLM interface
│   ├── tts.h                   # TTS interface
│   ├── stt_whisper.h           # Whisper STT implementation
│   ├── llm_llama.h             # Llama LLM implementation
│   ├── tts_paroli.h            # Paroli TTS implementation
│   └── config_manager.h        # Configuration management
├── src/                        # Source files
│   ├── main.cpp                # Main application entry point
│   ├── async_pipeline_factory.cpp # Pipeline factory implementation
│   ├── async_processors.cpp    # Processor implementations
│   ├── stt_whisper.cpp         # Whisper STT backend
│   ├── llm_llama.cpp           # Llama LLM backend
│   ├── tts_paroli.cpp          # Paroli TTS backend
│   ├── common.cpp              # Utility functions
│   └── common-sdl.cpp          # SDL audio utilities
├── scripts/                    # Utility scripts
│   └── setup.sh                # Setup script
├── config/                     # Configuration files
│   └── models.json             # Model paths and settings
├── models/                     # Model files (not in git)
│   ├── llm/                    # GGUF language models
│   ├── tts/                    # TTS ONNX models + config
│   └── stt/                    # Whisper models
├── third_party/                # External dependencies
│   ├── llama.cpp/              # Llama.cpp library
│   └── paroli-daemon/          # Paroli TTS library
├── deps/                       # Downloaded dependencies
│   ├── onnxruntime-*/          # ONNX Runtime
│   └── piper_phonemize/        # Piper phonemization
├── CMakeLists.txt              # CMake build configuration
├── Makefile                    # Convenient build targets
└── build/                      # Build artifacts (not in git)
```

## ⚙️ Configuration

### Configuration File
Edit `config/models.json` to customize:
- Model file paths
- Audio settings (sample rate, buffer size, VAD parameters)
- LLM parameters (context size, GPU layers, sampling)
- TTS voice settings

### Build Configuration
The project uses CMake with the following options:

**Backend Selection:**
- `-DUSE_WHISPER=ON` - Enable Whisper STT (default)
- `-DUSE_LLAMA=ON` - Enable Llama LLM (default)
- `-DUSE_Paroli=ON` - Enable Paroli TTS (default)
- `-DUSE_RKLLM=ON` - Enable RKLLM for embedded devices

**Build Options:**
- `-DCMAKE_BUILD_TYPE=Release` - Optimized release build (default)
- `-DCMAKE_BUILD_TYPE=Debug` - Debug build with symbols
- `-DENABLE_STATS_LOGGING=ON` - Enable performance statistics

## 🔧 Development

### Building
```bash
# Build optimized release version (default)
make build

# Build debug version with stats
make debug

# Clean build artifacts
make clean

# Show all available targets
make help
```

### Manual CMake Build
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make -j$(nproc)
```

### Performance Profiling
```bash
# Build with statistics logging
make debug

# Run with profiling
./build/local-llm
# Statistics will be printed on shutdown
```

### Adding New Backends
1. Create interface header in `include/` (e.g., `stt_new.h`)
2. Implement backend class in `src/` (e.g., `stt_new.cpp`)
3. Add CMake option in `CMakeLists.txt`
4. Update `async_pipeline_factory.cpp` to include new backend
5. Add conditional compilation with `#ifdef USE_NEW_BACKEND`

## 🐛 Troubleshooting

### Common Issues

**Audio Issues:**
- Ensure ALSA is properly configured: `aplay -l`
- Check microphone permissions
- Try different audio devices in config

**Performance Issues:**
- Enable GPU acceleration in Llama config
- Use optimized models (quantized GGUF)
- Build in Release mode: `make build`

**Build Issues:**
- Ensure all dependencies are installed
- Clean build: `make clean && make build`
- Check CMake version (3.16+ required)

### Debug Mode
```bash
# Build with debug symbols and stats
make debug

# Run with verbose output
./build/local-llm --config config/models.json
```

## 📊 Performance

### Optimizations Included
- **Compiler**: -O3 optimizations with native CPU targeting
- **Threading**: Parallel processing with hardware-concurrency awareness
- **Memory**: RAII, move semantics, smart pointers
- **Audio**: Low-latency ALSA with immediate interruption
- **GPU**: Optional GPU acceleration for LLM inference

### Expected Performance
- **Audio Latency**: <100ms end-to-end
- **STT Processing**: Real-time transcription
- **LLM Generation**: Depends on model size and hardware
- **TTS Synthesis**: Real-time audio generation

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes
4. Test with both debug and release builds
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) for STT
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for LLM
- [Paroli](https://github.com/paroli-ai/paroli-daemon) for TTS
- [ONNX Runtime](https://github.com/microsoft/onnxruntime) for TTS acceleration