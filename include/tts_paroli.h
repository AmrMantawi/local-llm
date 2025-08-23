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
    bool speak(const std::string &text) override;
    void shutdown() override;

private:
    bool initialized;
    std::string encoder_path;
    std::string decoder_path;
    std::string config_path;
    std::string espeak_data_path;
    std::unique_ptr<ParoliSynthesizer> synthesizer;
};
