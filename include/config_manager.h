#pragma once

#include <string>
#include <optional>

#ifdef DISABLE_JSON_CONFIG
// Stub implementation when JSON is disabled
class ConfigManager {
public:
    static ConfigManager& getInstance() {
        static ConfigManager instance;
        return instance;
    }
    
    bool loadConfig(const std::string& configPath) { return false; }
    
    std::string getModelPath(const std::string& category, const std::string& type) const {
        // Return default paths when JSON is disabled
        if (category == "stt" && type == "whisper") {
            return "../third_party/whisper.cpp/models/ggml-base.en.bin";
        } else if (category == "llm" && type == "llama") {
            return "../models/llm/jan-nano-4b-Q3_K_M.gguf";
        } else if (category == "tts" && type == "piper") {
            return "../models/tts/en_US-lessac-medium.onnx";
        }
        return "";
    }
    
    int getAudioSampleRate() const { return 16000; }
    int getAudioBufferMs() const { return 30000; }
    float getVadThreshold() const { return 0.6f; }
    int getVadCaptureMs() const { return 10000; }
    int getTtsVoiceId() const { return 2; }
    std::string getTtsSpeakScript() const { return "../scripts/speak"; }
    
private:
    ConfigManager() = default;
};
#else
// Full JSON implementation
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>

class ConfigManager {
public:
    static ConfigManager& getInstance() {
        static ConfigManager instance;
        return instance;
    }
    
    bool loadConfig(const std::string& configPath) {
        try {
            std::ifstream configFile(configPath);
            if (!configFile.is_open()) {
                std::cerr << "Failed to open config file: " << configPath << std::endl;
                return false;
            }
            
            configFile >> config;
            std::cout << "Configuration loaded from: " << configPath << std::endl;
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading config: " << e.what() << std::endl;
            return false;
        }
    }
    
    std::string getModelPath(const std::string& category, const std::string& type) const {
        try {
            return config["models"][category][type]["path"].get<std::string>();
        } catch (const std::exception& e) {
            std::cerr << "Error getting model path for " << category << "/" << type << ": " << e.what() << std::endl;
            return "";
        }
    }
    
    int getAudioSampleRate() const {
        try {
            return config["settings"]["audio"]["sample_rate"].get<int>();
        } catch (const std::exception& e) {
            return 16000; // default
        }
    }
    
    int getAudioBufferMs() const {
        try {
            return config["settings"]["audio"]["buffer_ms"].get<int>();
        } catch (const std::exception& e) {
            return 30000; // default
        }
    }
    
    float getVadThreshold() const {
        try {
            return config["settings"]["audio"]["vad_threshold"].get<float>();
        } catch (const std::exception& e) {
            return 0.6f; // default
        }
    }
    
    int getVadCaptureMs() const {
        try {
            return config["settings"]["audio"]["vad_capture_ms"].get<int>();
        } catch (const std::exception& e) {
            return 10000; // default
        }
    }
    
    int getTtsVoiceId() const {
        try {
            return config["settings"]["tts"]["voice_id"].get<int>();
        } catch (const std::exception& e) {
            return 2; // default
        }
    }
    
    std::string getTtsSpeakScript() const {
        try {
            return config["settings"]["tts"]["speak_script"].get<std::string>();
        } catch (const std::exception& e) {
            return "../scripts/speak"; // default
        }
    }
    
private:
    ConfigManager() = default;
    nlohmann::json config;
};
#endif 