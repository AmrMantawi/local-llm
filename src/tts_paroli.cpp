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
    encoder_path = config.getNestedModelPath("tts", "paroli", "encoder");
    decoder_path = config.getNestedModelPath("tts", "paroli", "decoder");
    config_path = config.getNestedModelPath("tts", "paroli", "config");
#ifdef ESPEAK_NG_DATA_DIR
    espeak_data_path = ESPEAK_NG_DATA_DIR;
    std::cout << "ESPEAK_NG_DATA_DIR defined as: " << ESPEAK_NG_DATA_DIR << std::endl;
#else
    std::cout << "ESPEAK_NG_DATA_DIR not defined" << std::endl;
#endif
    
    if(encoder_path.empty()) {
        std::cerr << "Paroli encoder model not found" << std::endl;
        return false;
    }

    if(decoder_path.empty()) {
        std::cerr << "Paroli decoder model not found" << std::endl;
        return false;
    }
    
    if(config_path.empty()) {
        std::cerr << "Paroli config file not found" << std::endl;
        return false;
    }
    
    if(espeak_data_path.empty()) {
        std::cerr << "Paroli espeak data not found" << std::endl;
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

bool TTSParoli::speakWithPhonemeTimings(const std::string &text, async_pipeline::AudioChunkMessage& audio_chunk, std::vector<PhonemeTimingInfo>& phoneme_timings) {
    if (!synthesizer) {
        std::cerr << "TTS not initialized" << std::endl;
        return false;
    }

    if (text.empty()) {
        return true;
    }

    try {
        // Generate complete audio for the text with timing information
        auto result = synthesizer->synthesizePcmWithTiming(text);
        
        if (result.audio.empty()) {
            std::cerr << "Failed to generate audio for text: " << text << std::endl;
            return false;
        }
        
        // Copy phoneme timing information
        phoneme_timings.clear();
        phoneme_timings.reserve(result.phoneme_timings.size());
        for (const auto& piper_phoneme : result.phoneme_timings) {
            PhonemeTimingInfo our_phoneme;
            our_phoneme.phoneme_id = piper_phoneme.phoneme_id;
            our_phoneme.duration_seconds = piper_phoneme.duration_seconds;
            phoneme_timings.push_back(our_phoneme);
        }
        
        // Write audio data to the provided AudioChunkMessage
        audio_chunk.audio_data = std::move(result.audio);
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
