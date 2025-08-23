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
        } else if (category == "tts" && type == "paroli_encoder") {
            return "../models/tts/paroli/encoder.onnx";
        } else if (category == "tts" && type == "paroli_decoder") {
            return "../models/tts/paroli/decoder.onnx";
        } else if (category == "tts" && type == "paroli_config") {
            return "../models/tts/paroli/config.json";
        } else if (category == "tts" && type == "paroli_espeak_data") {
            return "../models/tts/paroli/espeak-ng-data";
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
#include <filesystem>

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
            // Remember directory of the configuration to resolve relative model paths
            std::filesystem::path cfgPath(configPath);
            configDirectory_ = cfgPath.has_parent_path() ? cfgPath.parent_path().string() : std::string{"."};
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Error loading config: " << e.what() << std::endl;
            return false;
        }
    }
    
    std::string getModelPath(const std::string& category, const std::string& type) const {
        try {
            std::string pathFromConfig = config["models"][category][type]["path"].get<std::string>();
            std::filesystem::path p(pathFromConfig);
            if (p.is_relative() && !configDirectory_.empty()) {
                p = std::filesystem::path(configDirectory_) / p;
            }
            return p.string();
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
    std::string configDirectory_;
};
#endif 