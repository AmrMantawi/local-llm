#pragma once

#include <string>
#include <optional>

// JSON-based configuration manager
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
    
    std::string getNestedModelPath(const std::string& category, const std::string& backend, const std::string& component) const {
        try {
            std::string pathFromConfig = config["models"][category][backend][component]["path"].get<std::string>();
            std::filesystem::path p(pathFromConfig);
            if (p.is_relative() && !configDirectory_.empty()) {
                p = std::filesystem::path(configDirectory_) / p;
            }
            return p.string();
        } catch (const std::exception& e) {
            std::cerr << "Error getting nested model path for " << category << "/" << backend << "/" << component << ": " << e.what() << std::endl;
            return "";
        }
    }
    
    // Generalized backend-aware model getters
    std::string getSTTModelPath() const {
#ifdef USE_WHISPER
        return getModelPath("stt", "whisper");
#else
        std::cerr << "Error: No STT backend enabled at compile time" << std::endl;
        return "";
#endif
    }
    
    std::string getLLMModelPath() const {
#ifdef USE_LLAMA
        return getModelPath("llm", "llama");
#else
        std::cerr << "Error: No LLM backend enabled at compile time" << std::endl;
        return "";
#endif
    }
    
    std::string getTTSModelPath(const std::string& component = "") const {
#ifdef USE_Paroli
        if (component.empty()) {
            // Return encoder path as the primary model
            return getNestedModelPath("tts", "paroli", "encoder");
        } else {
            return getNestedModelPath("tts", "paroli", component);
        }
#else
        std::cerr << "Error: No TTS backend enabled at compile time" << std::endl;
        return "";
#endif
    }
    
    // Get all Paroli TTS model paths (for backends that need multiple files)
    struct ParoliPaths {
        std::string encoder;
        std::string decoder;
        std::string config;
    };
    
    ParoliPaths getParoliModelPaths() const {
        ParoliPaths paths;
#ifdef USE_Paroli
        paths.encoder = getNestedModelPath("tts", "paroli", "encoder");
        paths.decoder = getNestedModelPath("tts", "paroli", "decoder");
        paths.config = getNestedModelPath("tts", "paroli", "config");
#else
        std::cerr << "Error: Paroli TTS backend not enabled at compile time" << std::endl;
#endif
        return paths;
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
private:
    ConfigManager() = default;
    nlohmann::json config;
    std::string configDirectory_;
}; 