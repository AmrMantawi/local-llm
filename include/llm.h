#pragma once

#include <string>
#include <functional>

/// Interface for Large Language Models (LLMs)
class ILLM {
public:
    virtual ~ILLM() = default;

    /// Initialize the LLM.
    /// @return true on success, false on failure
    virtual bool init() = 0;

    /// Generate a response to the input prompt
    /// @param prompt Input text prompt
    /// @param response Final accumulated response
    /// @return true on success, false on failure
    virtual bool generate(const std::string &prompt, std::string &response) = 0;

    /// Async generate with text chunk callback for TTS streaming
    /// @param prompt Input text prompt
    /// @param response Final accumulated response
    /// @param callback Called for each complete text chunk generated
    /// @return true on success, false on failure
    virtual bool generate_async(const std::string &prompt, std::string &response, 
                               std::function<void(const std::string&)> callback) = 0;

    /// Release resources (optional cleanup)
    virtual void shutdown() = 0;
};
