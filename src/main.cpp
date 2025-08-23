// src/main.cpp

// Abstract interfaces
#include "stt.h"
#include "tts.h"
#include "config_manager.h"

// Whisper-specific headers
#ifdef USE_WHISPER
#include "common-sdl.h"
#include "common.h"
#include "stt_whisper.h"
#endif

// Llama-specific headers
#ifdef USE_LLAMA
#include "llm_llama.h"
#endif

// TTS-specific headers
#ifdef USE_Piper
#include "tts_piper.h"
#endif
#ifdef USE_Paroli
#include "tts_paroli.h"
#endif

#include <SDL2/SDL.h>
#include <iostream>
#include <vector>
#include <string>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include "server.h"

// Alias the selected backend classes
using STT = STT_BACKEND;
using LLM = LLM_BACKEND;
using TTS = TTS_BACKEND;

// Audio and VAD configuration
static const int    AUDIO_BUFFER_MS     = 30 * 1000;    // Total buffer size for audio_async 
static const int    AUDIO_SAMPLE_RATE   = 16000;        // Sample rate for audio capture
static const int    VAD_PRE_WINDOW_MS   = 2000;         // Duration to check for speech before full capture
static const int    VAD_CAPTURE_MS      = 10000;        // Duration of audio to capture after speech detected
static const int    VAD_START_MS        = 1250;         // Start analyzing after this duration
static const float  VAD_THRESHOLD       = 0.6f;         // Energy threshold for detecting voice
static const float  VAD_FREQ_CUTOFF     = 100.0f;       // Frequency cutoff for high-pass filter
static const bool   VAD_PRINT_ENERGY    = false;        // Toggle energy debug print

// Graceful shutdown on Ctrl+C
static std::atomic<bool> keep_running{true};

static void handle_sigint(int) 
{
    keep_running = false;
}

int main(int argc, char** argv) {
    // Check if the that STT_BACKEND, LLM_BACKEND, and TTS_BACKEND are defined
    static_assert(std::is_base_of<ISTT, STT>::value,
                  "STT_BACKEND must be a subclass of ISTT");
    static_assert(std::is_base_of<ILLM, LLM>::value,
                  "LLM_BACKEND must be a subclass of ILLM");
    static_assert(std::is_base_of<ITTS, TTS>::value,
                  "TTS_BACKEND must be a subclass of ITTS");

    // CLI flags: --config /path/models.json, --socket /tmp/local-llm.sock (accepted but not implemented)
    const char* configPath = "/usr/share/local-llm/config/models.json";
    std::string socketPath = "/run/local-llm.sock";
    std::string mode = "server"; // default server
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            configPath = argv[++i];
        } else if ((arg == "--socket" || arg == "-s") && i + 1 < argc) {
            socketPath = argv[++i];
        } else if ((arg == "--mode" || arg == "-m") && i + 1 < argc) {
            mode = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: local-llm [--mode server|mic] [--config /path/models.json] [--socket /run/local-llm.sock]\n";
            return 0;
        }
    }

    // Load configuration
    auto& config = ConfigManager::getInstance();
    if (!config.loadConfig(configPath)) {
        std::cout << "Using default configuration (config file not found or invalid)" << std::endl;
    }

    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);

    // Load LLM model path
    const std::string llama_model_path = config.getModelPath("llm", "llama");
    LLM llm;
    if (!llm.init(llama_model_path)) {
        std::cerr << "Failed to initialize LLM backend\n";
        return 1;
    }

    if (mode == "server") {
        // run unix socket server
        int rc = run_server(socketPath, llm, keep_running);
        llm.shutdown();
        return rc;
    }

    // mic mode: initialize STT and TTS
    // Initialize audio capture with configurable buffer size; retry to avoid startup races
    audio_async audio(config.getAudioBufferMs());
    {
        bool audio_initialized = false;
        for (int attempt = 1; attempt <= 8; ++attempt) {
            if (audio.init(-1, config.getAudioSampleRate())) { audio_initialized = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        if (!audio_initialized) {
            std::cerr << "audio.init() failed\n";
            return 1;
        }
    }
    audio.resume();
    std::cout << "Listening... (press Ctrl+C to stop)\n";

    const std::string whisper_model_path = config.getModelPath("stt", "whisper");
    STT stt;
    if (!stt.init(whisper_model_path)) {
        std::cerr << "Failed to initialize STT backend\n";
        return 1;
    }
    TTS tts;
    bool tts_initialized = false;
    if (!tts.init()) {
        std::cerr << "Failed to initialize TTS backend\n";
    } else {
        tts_initialized = true;
    }
    std::cout << "All backends initialized\n";

    std::vector<float> pcmf32;
    while (keep_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        audio.get(VAD_PRE_WINDOW_MS, pcmf32);
        if (::vad_simple(pcmf32, config.getAudioSampleRate(), VAD_START_MS, config.getVadThreshold(), VAD_FREQ_CUTOFF, VAD_PRINT_ENERGY))
        {
            audio.get(config.getVadCaptureMs(), pcmf32);

            std::string text;
            if (stt.transcribe(pcmf32, text) && !text.empty()) {
                std::cout << "→ " << text << "\n";
                std::string response;
                if (llm.generate(text, response)) {
                    std::cout << "BMO: " << response << "\n";
                    if (tts_initialized) {
                        if (!tts.speak(response)) {
                            std::cerr << "TTS failed to speak response\n";
                        }
                    }
                } else {
                    std::cerr << "LLM generation failed\n";
                }
            }
            audio.clear();
        }
    }

    // Cleanup
    stt.shutdown();
    llm.shutdown();
    if (tts_initialized) {
        tts.shutdown();
    }
    audio.pause();
    std::cout << "\nProgram Ended\n";
    return 0;
}