// src/main.cpp

#include "stt.h"

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

// Graceful shutdown on Ctrl+C
static std::atomic<bool> keep_running{true};
static void handle_sigint(int) {
    keep_running = false;
}

int main(int argc, char** argv) {
    static_assert(std::is_base_of<ISTT, WhisperSTT>::value,
                  "STT_BACKEND must be a subclass of ISTT");

    std::signal(SIGINT, handle_sigint);

    // Setup audio input
    audio_async audio(30 * 1000);  // 30 seconds

    if (!audio.init(-1, WHISPER_SAMPLE_RATE)) {
        std::cerr << "audio.init() failed\n";
        return 1;
    }

    audio.resume();
    std::cout << "Listening... (press Ctrl+C to stop)\n";

    // Load Whisper model
    const std::string model_path = "../third_party/whisper.cpp/models/ggml-small.en.bin";

    STT stt;
    if (!stt.init(model_path)) {
        std::cerr << "Failed to initialize STT backend\n";
        return 1;
    }

    std::cout << "STT backend initialized\n";

    // Transcription loop
    std::vector<float> pcmf32;
    while (keep_running) {
        // delay
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        audio.get(2000, pcmf32);  // fetch latest audio
        if (pcmf32.empty()) {
            SDL_Delay(100);
            continue;
        }

        std::string text;
        if (stt.transcribe(pcmf32, text) && !text.empty()) {
            std::cout << "→ " << text << "\n";
        }
    }

    stt.shutdown();
    audio.pause();

    std::cout << "\nProgram Ended\n";
    return 0;
}