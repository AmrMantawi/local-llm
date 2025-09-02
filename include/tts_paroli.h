#pragma once

#include "tts.h"
#include <string>
#include <memory>

// Forward declaration
class ParoliSynthesizer;

class TTSParoli : public ITTS {
public:
    TTSParoli();
    ~TTSParoli();
    
    bool init() override;
    bool speak(const std::string &text, async_pipeline::AudioChunkMessage& audio_chunk) override;

    // TODO: Add async speak with callback for streaming audio

    void shutdown() override;

private:
    std::string encoder_path;
    std::string decoder_path;
    std::string config_path;
    std::string espeak_data_path;
    std::unique_ptr<ParoliSynthesizer> synthesizer;
};
