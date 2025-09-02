# Local LLM

A complete local AI assistant with Speech-to-Text (STT), Large Language Model (LLM), and Text-to-Speech (TTS) capabilities. Run entirely on your machine with no cloud dependencies.

## 🚀 Features

- **Speech-to-Text**: Whisper-based transcription (configurable backends)
- **Language Model**: Llama-based text generation with GGUF model support (configurable backends)
- **Text-to-Speech**: Multiple TTS backends (Paroli, Piper) with ONNX support
- **Voice Activity Detection**: Automatic speech detection
- **Cross-platform**: Linux, macOS, Windows (WSL)
- **Privacy-first**: Everything runs locally
- **Modular**: Mix and match different backends for each component

## 📋 Requirements

### System Dependencies
- **CMake** 3.16+
- **SDL2** development libraries
- **Python** 3.7+
- **ALSA** (Linux) or **Core Audio** (macOS)
- **ONNX Runtime** (bundled with Paroli TTS)

### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install cmake libsdl2-dev python3 python3-pip python3-venv alsa-utils
```

### macOS
```bash
brew install cmake sdl2 python3
```

## 🛠️ Quick Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/amrmantawi/local-llm.git
   cd local-llm
   ```

2. **Run the setup script:**
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```

3. **Download models:**
   - **STT**: Download Whisper models from [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp)
   - **LLM**: Download GGUF models from [Hugging Face](https://huggingface.co/models?search=gguf)
   - **TTS**: 
     - **Paroli**: Download models (encoder.onnx, decoder.onnx, config.json) from [Paroli releases](https://github.com/paroli-ai/paroli-daemon/releases)
     - **Piper**: Download models from [Hugging Face](https://huggingface.co/rhasspy/piper-voices)

4. **Place models in the correct directories:**
   ```
   models/
   ├── stt/     # .bin files (Whisper)
   ├── llm/     # .gguf files (Llama)
   └── tts/     # .onnx files (Paroli/Piper) + config.json + piper_phonemize/
   ```

5. **Run the application:**
   ```bash
   # Microphone mode (interactive voice chat)
   ./build/local-llm --mode mic --config config/models.json
   
   # Server mode (Unix socket server)
   ./build/local-llm --mode server --config config/models.json
   ```

## 🔌 Backend Options

The project supports multiple backends for each component (STT, LLM, TTS). You can configure which backends to use during the build process.

### Speech-to-Text (STT) Backends

**Whisper (Default):**
- **Description**: High-quality speech recognition using OpenAI's Whisper model
- **Model Format**: `.bin` files (GGML format)
- **Build Option**: `-DUSE_WHISPER=ON`
- **Model Sources**: [Hugging Face](https://huggingface.co/ggerganov/whisper.cpp)
- **Usage**: `cmake -B build -DUSE_WHISPER=ON`

### Language Model (LLM) Backends

**Llama (Default):**
- **Description**: Large language model inference using llama.cpp
- **Model Format**: `.gguf` files (GGUF format)
- **Build Option**: `-DUSE_LLAMA=ON`
- **Model Sources**: [Hugging Face](https://huggingface.co/models?search=gguf)
- **Usage**: `cmake -B build -DUSE_LLAMA=ON`

**RKLLM (Alternative):**
- **Description**: Rockchip LLM backend for embedded devices
- **Model Format**: RKLLM-specific format
- **Build Option**: `-DUSE_RKLLM=ON`
- **Usage**: `cmake -B build -DUSE_RKLLM=ON -DUSE_LLAMA=OFF`

### Text-to-Speech (TTS) Backends

**Paroli (Default):**
- **Description**: ONNX-based neural TTS with high-quality voice synthesis
- **Model Format**: `.onnx` files + `config.json`
- **Build Option**: `-DUSE_Paroli=ON`
- **Model Sources**: [Paroli releases](https://github.com/paroli-ai/paroli-daemon/releases)
- **Dependencies**: ONNX Runtime, piper-phonemize
- **Usage**: `cmake -B build -DUSE_Paroli=ON -DUSE_Piper=OFF`

**Piper (Alternative):**
- **Description**: Fast neural TTS with good quality
- **Model Format**: `.onnx` files
- **Build Option**: `-DUSE_Piper=ON`
- **Model Sources**: [Hugging Face](https://huggingface.co/rhasspy/piper-voices)
- **Usage**: `cmake -B build -DUSE_Paroli=OFF -DUSE_Piper=ON`

### Example Build Configurations

**Default Configuration (Whisper + Llama + Paroli):**
```bash
cmake -B build -DUSE_WHISPER=ON -DUSE_LLAMA=ON -DUSE_Paroli=ON -DUSE_Piper=OFF
```

**Alternative Configuration (Whisper + Llama + Piper):**
```bash
cmake -B build -DUSE_WHISPER=ON -DUSE_LLAMA=ON -DUSE_Paroli=OFF -DUSE_Piper=ON
```

**Embedded Configuration (Whisper + RKLLM + Paroli):**
```bash
cmake -B build -DUSE_WHISPER=ON -DUSE_LLAMA=OFF -DUSE_RKLLM=ON -DUSE_Paroli=ON
```

## 🎯 Usage

### Microphone Mode
1. **Start the application** - it will initialize all backends
2. **Speak into your microphone** - voice activity detection will capture your speech
3. **Wait for transcription** - Whisper will convert speech to text
4. **Get AI response** - Llama will generate a response
5. **Hear the response** - Paroli will speak the response back to you

### Server Mode
The application can also run as a Unix socket server for integration with other applications.

**Controls:**
- Press `Ctrl+C` to exit gracefully

## 📁 Project Structure

```
local-llm/
├── include/           # Header files
│   ├── stt.h         # STT interface
│   ├── llm.h         # LLM interface
│   ├── tts.h         # TTS interface
│   └── ...           # Implementation headers
├── src/              # Source files
│   ├── main.cpp      # Main application
│   ├── stt_whisper.cpp
│   ├── llm_llama.cpp
│   ├── tts_paroli.cpp
│   ├── tts_piper.cpp
│   └── server.cpp
├── scripts/          # Utility scripts
│   ├── setup.sh      # Setup script
│   └── speak         # TTS script
├── config/           # Configuration files
│   └── models.json   # Model paths and settings
├── models/           # Model files (not in git)
│   ├── llm/         # GGUF language models
│   ├── tts/         # TTS ONNX models (Paroli/Piper) + config
│   └── stt/         # Whisper models
├── third_party/      # External dependencies
└── build/           # Build artifacts (not in git)
```

## ⚙️ Configuration

You can pass a custom config with `--config /path/to/models.json`. If omitted, the default is `/usr/share/local-llm/config/models.json`.

Edit `config/models.json` to customize:
- Model file paths
- Audio settings (sample rate, buffer size, VAD parameters)
- TTS voice settings

**Notes:**
- Relative model paths inside `models.json` are resolved relative to the config file's directory.
- The default configuration uses Paroli TTS with ONNX models.
- Model file paths
- Audio settings
- TTS voice settings

## 🔧 Development

### Building from source:
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### TTS Backend Options:
The project supports multiple TTS backends:

**Paroli (Default):**
```bash
cmake -B build -DUSE_Paroli=ON -DUSE_Piper=OFF
```

**Piper (Alternative):**
```bash
cmake -B build -DUSE_Paroli=OFF -DUSE_Piper=ON
```

### Adding new backends:
1. Create header file in `include/`
2. Create implementation in `src/`
3. Add CMake option and conditional compilation
4. Update main.cpp to include new backend

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- [whisper.cpp](https://github.com/ggerganov/whisper.cpp) for STT
- [llama.cpp](https://github.com/ggerganov/llama.cpp) for LLM
- [Paroli](https://github.com/paroli-ai/paroli-daemon) for TTS
- [Piper](https://github.com/rhasspy/piper) for TTS

