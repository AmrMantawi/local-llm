#include "pipeline_manager.h"
#include "config_manager.h"

// Backend includes
#ifdef USE_WHISPER
#include "stt_whisper.h"
#endif

#ifdef USE_LLAMA
#include "llm_llama.h"
#endif

#ifdef USE_Paroli
#include "tts_paroli.h"
#endif

namespace async_pipeline {

/**
 * Pipeline mode enumeration defining different use cases
 */
enum class PipelineMode {
    VOICE_ASSISTANT,    // Full pipeline: Audio → STT → LLM → TTS (microphone mode)
    TEXT_ONLY,          // LLM only: Text → LLM → Text (server/chat mode)
    TRANSCRIPTION,      // Audio → STT → Text (transcription service)
    SYNTHESIS           // Text → TTS → Audio (text-to-speech service)
};

/**
 * Factory functions to create backend instances
 */
class PipelineFactory {
public:
    static std::unique_ptr<ISTT> create_stt_backend() {
#ifdef USE_WHISPER
        auto stt = std::make_unique<WhisperSTT>();
        auto& config = ConfigManager::getInstance();
        const std::string whisper_model_path = config.getSTTModelPath();
        
        if (!stt->init(whisper_model_path)) {
            std::cerr << "[PipelineFactory] Failed to initialize Whisper STT backend" << std::endl;
            return nullptr;
        }
        
        return stt;
#else
        std::cerr << "[PipelineFactory] Whisper STT backend not available (USE_WHISPER not defined)" << std::endl;
        return nullptr;
#endif
    }
    
    static std::unique_ptr<ILLM> create_llm_backend() {
#ifdef USE_LLAMA
        // Create LLM backend without initializing (processor will handle init)
        auto llm = std::make_unique<LlamaLLM>();
        return llm;
#else
        std::cerr << "[PipelineFactory] Llama LLM backend not available (USE_LLAMA not defined)" << std::endl;
        return nullptr;
#endif
    }
    
    static std::unique_ptr<ITTS> create_tts_backend() {
#ifdef USE_Paroli
        auto tts = std::make_unique<TTSParoli>();
        return tts;
#else
        std::cerr << "[PipelineFactory] No TTS backend available (USE_Paroli not defined)" << std::endl;
        return nullptr;
#endif
    }
    
    /**
     * Create pipeline based on specified mode
     */
    static std::unique_ptr<PipelineManager> create_pipeline(PipelineMode mode = PipelineMode::VOICE_ASSISTANT, bool enable_stats = false);
    

};

// Factory method implementation
std::unique_ptr<PipelineManager> PipelineFactory::create_pipeline(PipelineMode mode, bool enable_stats) {
    PipelineConfig config;
    
    // Configure components based on mode
    switch (mode) {
        case PipelineMode::VOICE_ASSISTANT:
            // Full pipeline: Audio → STT → LLM → TTS
            config.enable_stt = true;
            config.enable_llm = true;
            config.enable_tts = true;
            break;
            
        case PipelineMode::TEXT_ONLY:
            // LLM only: Text → LLM → Text
            config.enable_stt = false;
            config.enable_llm = true;
            config.enable_tts = false;
            break;
            
        case PipelineMode::TRANSCRIPTION:
            // Audio → STT → Text
            config.enable_stt = true;
            config.enable_llm = false;
            config.enable_tts = false;
            break;
            
        case PipelineMode::SYNTHESIS:
            // Text → TTS → Audio
            config.enable_stt = false;
            config.enable_llm = false;
            config.enable_tts = true;
            break;
    }
    
    // Stats logging is independent of mode
    config.enable_stats_logging = enable_stats;
    
    // Create pipeline with the configured settings
    auto pipeline = std::make_unique<PipelineManager>(config);
    
    // Create backends
    auto stt_backend = config.enable_stt ? PipelineFactory::create_stt_backend() : nullptr;
    auto llm_backend = config.enable_llm ? PipelineFactory::create_llm_backend() : nullptr;
    auto tts_backend = config.enable_tts ? PipelineFactory::create_tts_backend() : nullptr;
    
    // Initialize pipeline
    if (!pipeline->initialize(std::move(stt_backend), std::move(llm_backend), std::move(tts_backend))) {
        std::cerr << "[PipelineFactory] Failed to initialize pipeline" << std::endl;
        return nullptr;
    }
    
    return pipeline;
}



} // namespace async_pipeline
