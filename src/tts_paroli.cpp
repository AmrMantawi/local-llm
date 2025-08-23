#include "tts_paroli.h"
#include "config_manager.h"
#include "paroli_daemon.hpp"
#include <iostream>
#include <filesystem>

TTSParoli::TTSParoli() 
    : initialized(false) {
}

TTSParoli::~TTSParoli() {
    shutdown();
}

bool TTSParoli::init() {
    // Initialize from configuration
    auto& config = ConfigManager::getInstance();
    
    // Get model paths from config
    encoder_path = config.getModelPath("tts", "paroli_encoder");
    decoder_path = config.getModelPath("tts", "paroli_decoder");
    config_path = config.getModelPath("tts", "paroli_config");
    espeak_data_path = config.getModelPath("tts", "paroli_espeak_data");
    
    // Check if model files exist
    if (!std::filesystem::exists(encoder_path)) {
        std::cerr << "Paroli encoder model not found at: " << encoder_path << std::endl;
        return false;
    }
    
    if (!std::filesystem::exists(decoder_path)) {
        std::cerr << "Paroli decoder model not found at: " << decoder_path << std::endl;
        return false;
    }
    
    if (!std::filesystem::exists(config_path)) {
        std::cerr << "Paroli config file not found at: " << config_path << std::endl;
        return false;
    }
    
    if (!std::filesystem::exists(espeak_data_path)) {
        std::cerr << "Paroli espeak data not found at: " << espeak_data_path << std::endl;
        return false;
    }
    
    try {
        // Initialize ParoliSynthesizer
        ParoliSynthesizer::InitOptions opts;
        opts.encoderPath = encoder_path;
        opts.decoderPath = decoder_path;
        opts.modelConfigPath = config_path;
        opts.eSpeakDataPath = espeak_data_path;
        opts.accelerator = ""; // Use CPU by default
        
        synthesizer = std::make_unique<ParoliSynthesizer>(opts);
        
        if (!synthesizer->isInitialized()) {
            std::cerr << "Failed to initialize ParoliSynthesizer: " << synthesizer->getLastError() << std::endl;
            return false;
        }
        
        // Set volume to 0.8 (80%)
        synthesizer->setVolume(0.8f);
        
        initialized = true;
        std::cout << "TTS (Paroli) initialized\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during ParoliSynthesizer initialization: " << e.what() << std::endl;
        return false;
    }
}

bool TTSParoli::speak(const std::string &text) {
    if (!initialized || !synthesizer) {
        std::cerr << "TTS not initialized" << std::endl;
        return false;
    }

    if (text.empty()) {
        return true;
    }

    try {
        // Use the speak method which directly outputs to speakers
        bool success = synthesizer->speak(text);
        if (!success) {
            std::cerr << "Paroli speak failed: " << synthesizer->getLastError() << std::endl;
        }
        return success;
    } catch (const std::exception& e) {
        std::cerr << "Exception during speech synthesis: " << e.what() << std::endl;
        return false;
    }
}

void TTSParoli::shutdown() {
    initialized = false;
    synthesizer.reset();
}
