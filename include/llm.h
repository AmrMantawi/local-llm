#pragma once

#include <string>

/// Interface for Local Language Models (LLMs)
class ILLM {
public:
    virtual ~ILLM() = default;

    /// Initialize the LLM with a given model path
    virtual bool init(const std::string &modelPath) = 0;

    /// Generate a response to the input prompt
    virtual bool generate(const std::string &prompt, std::string &response) = 0;

    /// Release resources (optional cleanup)
    virtual void shutdown() = 0;
};
