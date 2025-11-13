#pragma once

#include <string>
#include <optional>

// JSON-based configuration manager
#include <nlohmann/json.hpp>
#include <fstream>
#include <iostream>
#include <filesystem>
#include <stdexcept>

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
    
    std::string getNestedModelPath(const std::string& category, const std::string& backend, const std::string& component) const {
        try {
            std::string pathFromConfig = config["models"][category][backend][component]["path"].get<std::string>();
            std::filesystem::path p(pathFromConfig);
            if (p.is_relative() && !configDirectory_.empty()) {
                p = std::filesystem::path(configDirectory_) / p;
            }
            if (!std::filesystem::exists(p)) {
                throw std::runtime_error("Model component not found at: " + p.string());
            }
            return p.string();
        } catch (const std::exception& e) {
            std::cerr << "Error getting nested model path for " << category << "/" << backend << "/" << component << ": " << e.what() << std::endl;
            throw;
        }
    }
    
    std::string getAudioDevice() const {
        try {
            return config["settings"]["audio"]["alsa_device"].get<std::string>();
        } catch (const std::exception&) {
            return "default";
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
private:
    ConfigManager() = default;
    nlohmann::json config;
    std::string configDirectory_;
}; 