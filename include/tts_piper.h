// include/tts_piper.h
#pragma once

#include "tts.h"
#include <string>

/// Piper-based TTS adapter.
class TTSPiper : public ITTS {
public:
    TTSPiper() = default;

    /// Initialize the Piper TTS system.
    /// @return true on success, false on failure.
    bool init() override;

    /// Speak the given text using Piper.
    /// @param text UTF-8 text to speak.
    /// @return true on success, false on failure.
    bool speak(const std::string &text) override;

    /// Release any resources held by Piper TTS.
    void shutdown() override;

private:
    bool initialized = false;
    std::string speak_script;
    std::string speak_file = "./to_speak.txt";
    int voice_id;
}; 