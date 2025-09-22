#pragma once

#include <memory>

namespace async_pipeline {

// Pipeline mode enumeration defining different use cases
enum class PipelineMode {
    VOICE_ASSISTANT,                // Audio → STT → LLM → TTS
    TEXT_ONLY,                      // LLM only: Text → LLM → Text
    TRANSCRIPTION,                  // Audio → STT → Text
    SYNTHESIS,                      // Text → TTS → Audio
    VOICE_ASSISTANT_WITH_ALT_TEXT   // Full pipeline with alternate text input/output enabled
};

class PipelineManager;

class PipelineFactory {
public:
    static std::unique_ptr<PipelineManager> create_pipeline(PipelineMode mode = PipelineMode::VOICE_ASSISTANT);
};

} // namespace async_pipeline


