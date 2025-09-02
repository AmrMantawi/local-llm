#pragma once

#include "async_pipeline.h"
#include "async_processors.h"
#include "stt.h"
#include "llm.h"
#include "tts.h"
#include <memory>
#include <map>
#include <chrono>

namespace async_pipeline {

/**
 * Configuration for the async pipeline
 */
struct PipelineConfig {
    // Queue sizes

    size_t text_queue_size = 20;
    size_t response_queue_size = 20;
    size_t control_queue_size = 100;  // High priority, needs to be responsive
    
    // Timeouts (milliseconds)
    int audio_timeout_ms = 1000;
    int text_timeout_ms = 500;
    int response_timeout_ms = 1000;
    
    // Enable/disable components
    bool enable_stt = true;
    bool enable_llm = true;
    bool enable_tts = true;
    
    // Monitoring
    bool enable_stats_logging = false;
    int stats_log_interval_seconds = 10;
};

/**
 * Pipeline manager that coordinates all processors and handles lifecycle
 */
class PipelineManager {
public:
    explicit PipelineManager(const PipelineConfig& config = PipelineConfig{})
        : config_(config), running_(false) {}
    
    ~PipelineManager() {
        stop();
    }
    
    /**
     * Initialize the pipeline with backend implementations
     */
    bool initialize(std::unique_ptr<ISTT> stt_backend,
                   std::unique_ptr<ILLM> llm_backend,
                   std::unique_ptr<ITTS> tts_backend) {
        if (running_) {
            std::cerr << "[PipelineManager] Cannot initialize while running" << std::endl;
            return false;
        }
        
        try {
            // Create queues
            text_queue_ = std::make_unique<SafeQueue<TextMessage>>(config_.text_queue_size);
            response_queue_ = std::make_unique<SafeQueue<TextMessage>>(config_.response_queue_size);
            control_queue_ = std::make_unique<SafeQueue<ControlMessage>>(config_.control_queue_size);
            
            // Create processors
            if (config_.enable_stt && stt_backend) {
                stt_processor_ = std::make_unique<STTProcessor>(*text_queue_, *control_queue_, std::move(stt_backend));
            }
            
            if (config_.enable_llm && llm_backend) {
                llm_processor_ = std::make_unique<LLMProcessor>(*text_queue_, *response_queue_, *control_queue_, std::move(llm_backend));
            }
            
            if (config_.enable_tts && tts_backend) {
                tts_processor_ = std::make_unique<TTSProcessor>(*response_queue_, *control_queue_, std::move(tts_backend));
            }
            
            std::cout << "[PipelineManager] Initialized successfully" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "[PipelineManager] Initialization failed: " << e.what() << std::endl;
            cleanup();
            return false;
        }
    }
    
    /**
     * Start the pipeline (all enabled processors)
     */
    bool start() {
        if (running_) {
            std::cerr << "[PipelineManager] Pipeline already running" << std::endl;
            return false;
        }
        
        if (!stt_processor_ && !llm_processor_ && !tts_processor_) {
            std::cerr << "[PipelineManager] No processors to start" << std::endl;
            return false;
        }
        
        try {
            // Start processors in reverse order (TTS first, Audio last)
            // This ensures downstream processors are ready before upstream ones start producing
            if (tts_processor_ && !tts_processor_->start()) {
                throw std::runtime_error("Failed to start TTS processor");
            }
            
            if (llm_processor_ && !llm_processor_->start()) {
                throw std::runtime_error("Failed to start LLM processor");
            }
            
            if (stt_processor_ && !stt_processor_->start()) {
                throw std::runtime_error("Failed to start STT processor");
            }
            
            running_ = true;
            
            // TODO: Create flag to enable/disable stat monitoring
            // Start monitoring thread if enabled
            if (config_.enable_stats_logging) {
                monitoring_thread_ = std::thread(&PipelineManager::monitor_loop, this);
            }
            
            std::cout << "[PipelineManager] Pipeline started successfully" << std::endl;
            return true;
            
        } catch (const std::exception& e) {
            std::cerr << "[PipelineManager] Failed to start pipeline: " << e.what() << std::endl;
            stop();
            return false;
        }
    }
    
    /**
     * Stop the pipeline gracefully
     */
    void stop() {
        if (!running_) return;
        
        std::cout << "[PipelineManager] Stopping pipeline..." << std::endl;
        running_ = false;
        
        // Shutdown queues first to wake up any blocking processors
        if (text_queue_) text_queue_->shutdown();
        if (response_queue_) response_queue_->shutdown();
        if (control_queue_) control_queue_->shutdown();
        
        // Now stop processors in forward order (STT first, TTS last)
        if (stt_processor_) {
            stt_processor_->stop();
        }
        
        if (llm_processor_) {
            llm_processor_->stop();
        }
        
        if (tts_processor_) {
            tts_processor_->stop();
        }
        
        // TODO: Add flag to enable/disable stat monitoring
        // Stop monitoring
        if (monitoring_thread_.joinable()) {
            monitoring_thread_.join();
        }
        
        std::cout << "[PipelineManager] Pipeline stopped" << std::endl;
    }
    
    /**
     * Check if pipeline is running
     */
    bool is_running() const {
        return running_;
    }
    
    /**
     * Get overall pipeline statistics
     */
    struct PipelineStats {
        BaseProcessor::Stats stt_stats;
        BaseProcessor::Stats llm_stats;
        BaseProcessor::Stats tts_stats;
        

        size_t text_queue_size = 0;
        size_t response_queue_size = 0;
        size_t control_queue_size = 0;
        
        std::chrono::steady_clock::time_point last_update;
    };
    
    PipelineStats get_stats() const {
        PipelineStats stats;
    
        if (stt_processor_) stats.stt_stats = stt_processor_->get_stats();
        if (llm_processor_) stats.llm_stats = llm_processor_->get_stats();
        if (tts_processor_) stats.tts_stats = tts_processor_->get_stats();

        if (text_queue_) stats.text_queue_size = text_queue_->size();
        if (response_queue_) stats.response_queue_size = response_queue_->size();
        if (control_queue_) stats.control_queue_size = control_queue_->size();
        
        stats.last_update = std::chrono::steady_clock::now();
        
        return stats;
    }
    
    /**
     * Process a single text input (bypasses audio/STT for server mode)
     */
    bool process_text_input(const std::string& text, std::string& response) {
        if (!running_ || !llm_processor_) {
            return false;
        }
        
        // Create text message and push to LLM queue
        TextMessage text_msg(text);
        if (!text_queue_->push(std::move(text_msg), std::chrono::milliseconds(config_.text_timeout_ms))) {
            return false;
        }
        
        // Wait for response
        TextMessage llm_response;
        if (!response_queue_->pop(llm_response, std::chrono::milliseconds(config_.response_timeout_ms))) {
            return false;
        }
        
        response = llm_response.text;
        return true;
    }
    
    /**
     * Clear all queues
     */
    void clear_queues() {

        if (text_queue_) text_queue_->clear();
        if (response_queue_) response_queue_->clear();
        if (control_queue_) control_queue_->clear();
    }

private:
    PipelineConfig config_;
    std::atomic<bool> running_;

    
    // Queues
    std::unique_ptr<SafeQueue<TextMessage>> text_queue_;
    std::unique_ptr<SafeQueue<TextMessage>> response_queue_;
    std::unique_ptr<SafeQueue<ControlMessage>> control_queue_;
    
    // Processors
    std::unique_ptr<STTProcessor> stt_processor_;
    std::unique_ptr<LLMProcessor> llm_processor_;
    std::unique_ptr<TTSProcessor> tts_processor_;
    
    // Monitoring
    std::thread monitoring_thread_;
    
    void cleanup() {
        stt_processor_.reset();
        llm_processor_.reset();
        tts_processor_.reset();
        
        text_queue_.reset();
        response_queue_.reset();
        control_queue_.reset();
    }
    
    void monitor_loop() {
        while (running_) {
            std::this_thread::sleep_for(std::chrono::seconds(config_.stats_log_interval_seconds));
            
            if (!running_) break;
            
            auto stats = get_stats();
            log_stats(stats);
        }
    }
    
    void log_stats(const PipelineStats& stats) {
        std::cout << "\n[PipelineManager] === Pipeline Statistics ===" << std::endl;
        
        // Audio processing is now integrated into STT processor
        
        if (stt_processor_) {
            std::cout << "[STT] Processed: " << stats.stt_stats.messages_processed
                      << ", Avg Time: " << stats.stt_stats.avg_processing_time.count() << "ms" << std::endl;
        }
        
        if (llm_processor_) {
            std::cout << "[LLM] Processed: " << stats.llm_stats.messages_processed
                      << ", Avg Time: " << stats.llm_stats.avg_processing_time.count() << "ms" << std::endl;
        }
        
        if (tts_processor_) {
            std::cout << "[TTS] Processed: " << stats.tts_stats.messages_processed
                      << ", Avg Time: " << stats.tts_stats.avg_processing_time.count() << "ms" << std::endl;
        }
        
        std::cout << "[Queues] Text: " << stats.text_queue_size 
                  << ", Text: " << stats.text_queue_size 
                  << ", Response: " << stats.response_queue_size 
                  << ", Control: " << stats.control_queue_size << std::endl;
        std::cout << "==========================================\n" << std::endl;
    }
};

} // namespace async_pipeline
