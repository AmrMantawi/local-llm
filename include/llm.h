#pragma once

#include <string>
#include <functional>

/// Interface for Local Language Models (LLMs)
class ILLM {
public:
    virtual ~ILLM() = default;

    /// Initialize the LLM with a given model path
    /// @param modelPath Path to the model file
    /// @return true on success, false on failure
    virtual bool init(const std::string &modelPath) = 0;

    /// Generate a response to the input prompt
    /// @param prompt Input text prompt
    /// @param response Final accumulated response
    /// @return true on success, false on failure
    virtual bool generate(const std::string &prompt, std::string &response) = 0;

    /// Async generate with sentence-by-sentence callback for TTS streaming
    /// @param prompt Input text prompt
    /// @param response Final accumulated response
    /// @param sentence_callback Called for each complete sentence generated
    /// @return true on success, false on failure
    virtual bool generate_async(const std::string &prompt, std::string &response, 
                               std::function<void(const std::string&)> sentence_callback) = 0;

    /// Release resources (optional cleanup)
    virtual void shutdown() = 0;
};
