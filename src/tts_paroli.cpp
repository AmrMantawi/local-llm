#include "tts_paroli.h"
#include "config_manager.h"
#include "paroli_daemon.hpp"
#include "async_pipeline.h"
#include <iostream>
#include <filesystem>
#include <vector>

TTSParoli::TTSParoli() {
}

TTSParoli::~TTSParoli() {
    shutdown();
}

bool TTSParoli::init() {
    // Initialize from configuration
    auto& config = ConfigManager::getInstance();
    
    // Get model paths from config using generalized method
    auto paroli_paths = config.getParoliModelPaths();
    encoder_path = paroli_paths.encoder;
    decoder_path = paroli_paths.decoder;
    config_path = paroli_paths.config;
    espeak_data_path = "deps/piper_phonemize/share/espeak-ng-data";
    
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
        
        std::cout << "TTS (Paroli) initialized\n";
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception during ParoliSynthesizer initialization: " << e.what() << std::endl;
        return false;
    }
}

bool TTSParoli::speak(const std::string &text, async_pipeline::AudioChunkMessage& audio_chunk) {
    if (!synthesizer) {
        std::cerr << "TTS not initialized" << std::endl;
        return false;
    }

    if (text.empty()) {
        return true;
    }

    try {
        // Generate complete audio for the text
        std::vector<int16_t> audio_data = synthesizer->synthesizePcm(text);
        
        if (audio_data.empty()) {
            std::cerr << "Failed to generate audio for text: " << text << std::endl;
            return false;
        }
        
        // Write audio data to the provided AudioChunkMessage
        audio_chunk.audio_data = std::move(audio_data);
        audio_chunk.sample_rate = synthesizer->nativeSampleRate();
        
        return true;
        
    } catch (const std::exception& e) {
        std::cerr << "TTS synthesis error: " << e.what() << std::endl;
        return false;
    }
}

void TTSParoli::shutdown() {
    if (!synthesizer) {
        return; // Already shut down
    }
    
    // Nothing to interrupt since synthesis is synchronous
    
    synthesizer.reset();
}
