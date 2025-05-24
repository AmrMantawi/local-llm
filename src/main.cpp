// src/main.cpp

// Abstract interfaces
#include "stt.h"

// Whisper-specific headers
#ifdef USE_WHISPER
#include "common-sdl.h"
#include "common.h"
#include "stt_whisper.h"
#endif

#include <SDL2/SDL.h>
#include <iostream>
#include <vector>
#include <string>
#include <csignal>
#include <atomic>

// Alias the selected backend class to STTBackend
using STT = STT_BACKEND;

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
    // Check if the that the STT_BACKEND is defined correctly
    static_assert(std::is_base_of<ISTT, STT>::value,
                  "STT_BACKEND must be a subclass of ISTT");

    std::signal(SIGINT, handle_sigint);

    // Initialize audio capture with a 30-second internal buffer
    audio_async audio(AUDIO_BUFFER_MS);

    // Try to open the default microphone
    if (!audio.init(-1, AUDIO_SAMPLE_RATE)) {
        std::cerr << "audio.init() failed\n";
        return 1;
    }

    // Start capturing audio
    audio.resume();
    std::cout << "Listening... (press Ctrl+C to stop)\n";

    // Load Whisper model
    const std::string model_path = "../third_party/whisper.cpp/models/ggml-base.en.bin";

    // Create and initialize STT backend
    STT stt;
    if (!stt.init(model_path)) {
        std::cerr << "Failed to initialize STT backend\n";
        return 1;
    }

    std::cout << "STT backend initialized\n";

    // Transcription loop
    std::vector<float> pcmf32;
    while (keep_running) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); // Small sleep to reduce CPU usage

        audio.get(VAD_PRE_WINDOW_MS, pcmf32);  // Fetch latest audio

        // Run simple voice activity detection to decide if someone is speaking
        if (::vad_simple(pcmf32, AUDIO_SAMPLE_RATE, VAD_START_MS, VAD_THRESHOLD, VAD_FREQ_CUTOFF, VAD_PRINT_ENERGY))
        {
            audio.get(VAD_CAPTURE_MS, pcmf32);  // Fetch detected speech

            // Transcribe audio using STT backend
            std::string text;
            if (stt.transcribe(pcmf32, text) && !text.empty()) {
                std::cout << "→ " << text << "\n";
            }

            audio.clear();
        }
    }

    // Cleanup
    stt.shutdown();
    audio.pause();

    std::cout << "\nProgram Ended\n";
    return 0;
}