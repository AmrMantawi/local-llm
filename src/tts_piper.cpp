#include "tts.h"
#include <iostream>
#include <cstdlib>
#include <fstream>
#include <filesystem>

class TTSPiper : public ITTS {
private:
    bool initialized = false;
    std::string speak_script = "../speak";
    std::string speak_file = "./to_speak.txt";
    int voice_id = 2;

public:
    bool init() override {
        // Check if piper is available in virtual environment first (relative to build directory)
        int result = system("test -f ../venv/bin/piper");
        if (result == 0) {
            initialized = true;
            std::cout << "TTS (Piper) initialized (using virtual environment)\n";
            return true;
        }
        
        // Check if piper is available system-wide
        result = system("which piper > /dev/null 2>&1");
        if (result != 0) {
            std::cerr << "Piper not found. Please install piper: pip install piper-tts\n";
            return false;
        }
        
        // Check if aplay is available (needed for audio playback)
        result = system("which aplay > /dev/null 2>&1");
        if (result != 0) {
            std::cerr << "aplay not found. Please install alsa-utils: sudo apt-get install alsa-utils\n";
            return false;
        }
        
        initialized = true;
        std::cout << "TTS (Piper) initialized (using system installation)\n";
        return true;
    }

    bool speak(const std::string &text) override {
        if (!initialized) {
            std::cerr << "TTS not initialized\n";
            return false;
        }

        if (text.empty()) {
            return true;
        }

        // Write text to file
        std::ofstream file(speak_file);
        if (!file.is_open()) {
            std::cerr << "Failed to open speak file: " << speak_file << "\n";
            return false;
        }
        file << text;
        file.close();

        // Build the speak command (similar to talk-llama.cpp)
        std::string command = speak_script + " " + std::to_string(voice_id) + " " + speak_file + " 2>/dev/null";
        
        int result = system(command.c_str());
        return result == 0;
    }

    void shutdown() override {
        initialized = false;
        
        // Clean up the speak file
        if (std::filesystem::exists(speak_file)) {
            std::filesystem::remove(speak_file);
        }
    }
};

// Define the TTS backend to use
using TTS_BACKEND = TTSPiper;
