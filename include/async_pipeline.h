#pragma once

#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <chrono>
#include <memory>
#include <functional>
#include <iostream>

namespace async_pipeline {

// Forward declarations
class STTProcessor;
class LLMProcessor;
class TTSProcessor;
class PipelineManager;

struct TextMessage {
    std::string text;
    // Stats
    // std::chrono::steady_clock::time_point timestamp;
    // float confidence;
    
    TextMessage() {}
    TextMessage(std::string txt)
        : text(std::move(txt)) {}
};

struct AudioChunkMessage {
    std::vector<int16_t> audio_data;
    unsigned int sample_rate;
    
    AudioChunkMessage() : sample_rate(22050) {}
    AudioChunkMessage(std::vector<int16_t> audio, unsigned int rate = 22050)
        : audio_data(std::move(audio)), sample_rate(rate) {}
};

/**
 * Control message for pipeline coordination (interruption, flush, etc.)
 */
struct ControlMessage {
    enum Type {
        INTERRUPT,      // Interrupt current processing
        FLUSH_QUEUES,   // Flush all downstream queues
        PAUSE,          // Pause processing
        RESUME,         // Resume processing
        SHUTDOWN        // Shutdown the entire pipeline
    };
    
    Type type;

    // Stats
    // std::chrono::steady_clock::time_point timestamp;
    
    ControlMessage(Type t) 
        : type(t) {}
};

/**
 * Thread-safe queue implementation for inter-component communication
 */
template<typename T>
class SafeQueue {
public:
    explicit SafeQueue(size_t max_size = 100) : max_size_(max_size), shutdown_(false) {}
    
    ~SafeQueue() {
        shutdown();
    }
    
    // Push with timeout - returns false if queue is full and timeout exceeded
    bool push(T item, std::chrono::milliseconds timeout = std::chrono::milliseconds(1000)) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (shutdown_) return false;
        
        // Wait for space or timeout
        if (!not_full_.wait_for(lock, timeout, [this] { 
            return queue_.size() < max_size_ || shutdown_; 
        })) {
            return false; // Timeout
        }
        
        if (shutdown_) return false;
        
        queue_.push(std::move(item));
        not_empty_.notify_one();
        return true;
    }
    
    // Blocking push - waits indefinitely until space available or shutdown
    bool push_blocking(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (shutdown_) return false;
        
        not_full_.wait(lock, [this] { 
            return queue_.size() < max_size_ || shutdown_; 
        });
        
        if (shutdown_) return false;
        
        queue_.push(std::move(item));
        not_empty_.notify_one();
        return true;
    }

    // Pop with timeout - returns false if queue is empty and timeout exceeded
    bool pop(T& item, std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (!not_empty_.wait_for(lock, timeout, [this] { 
            return !queue_.empty() || shutdown_; 
        })) {
            return false; // Timeout
        }
        
        if (shutdown_ && queue_.empty()) return false;
        
        item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return true;
    }
    
    // Blocking pop - waits indefinitely until item available or shutdown
    bool pop_blocking(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        not_empty_.wait(lock, [this] { 
            return !queue_.empty() || shutdown_; 
        });
        
        if (shutdown_ && queue_.empty()) return false;
        
        item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return true;
    }
    
    // Non-blocking pop
    bool try_pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty() || shutdown_) return false;
        
        item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return true;
    }
    
    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
    
    bool empty() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.empty();
    }
    
    void clear() {
        std::lock_guard<std::mutex> lock(mutex_);
        std::queue<T> empty;
        queue_.swap(empty);
        not_full_.notify_all();
    }
    
    // Flush all items and return count of flushed items
    size_t flush() {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t count = queue_.size();
        std::queue<T> empty;
        queue_.swap(empty);
        not_full_.notify_all();
        return count;
    }
    
    void shutdown() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            shutdown_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

private:
    mutable std::mutex mutex_;
    std::queue<T> queue_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    size_t max_size_;
    std::atomic<bool> shutdown_;
};

/**
 * Base processor class with common thread management
 */
class BaseProcessor {
public:
    BaseProcessor(const std::string& name) : name_(name), running_(false) {}
    virtual ~BaseProcessor() { stop(); }
    
    // Start the processor thread
    virtual bool start() {
        if (running_) return false;
        
        if (!initialize()) {
            return false;
        }
        
        running_ = true;
        thread_ = std::thread(&BaseProcessor::run, this);
        return true;
    }
    
    // Stop the processor thread
    virtual void stop() {
        if (!running_) return;
        
        running_ = false;
        if (thread_.joinable()) {
            thread_.join();
        }
        cleanup();
    }
    
    bool is_running() const { return running_; }
    const std::string& name() const { return name_; }
    
    // Interruption support
    virtual void interrupt() {
        std::lock_guard<std::mutex> lock(interrupt_mutex_);
        interrupt_requested_ = true;
    }
    
    virtual void clear_interrupt() {
        std::lock_guard<std::mutex> lock(interrupt_mutex_);
        interrupt_requested_ = false;
    }
    
    bool is_interrupt_requested() const {
        std::lock_guard<std::mutex> lock(interrupt_mutex_);
        return interrupt_requested_;
    }
    
    // Get processing statistics
    struct Stats {
        uint64_t messages_processed = 0;
        std::chrono::milliseconds avg_processing_time{0};
        std::chrono::steady_clock::time_point last_activity;
    };
    
    Stats get_stats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return stats_;
    }

protected:
    // Override in derived classes
    virtual bool initialize() = 0;
    virtual void process() = 0;
    virtual void cleanup() {}
    
    // Main thread loop
    void run() {
        while (running_) {
            try {
                auto start_time = std::chrono::steady_clock::now();
                process();
                auto end_time = std::chrono::steady_clock::now();
                
                // Update statistics
                {
                    std::lock_guard<std::mutex> lock(stats_mutex_);
                    stats_.messages_processed++;
                    stats_.last_activity = end_time;
                    
                    auto processing_time = std::chrono::duration_cast<std::chrono::milliseconds>(
                        end_time - start_time);
                    stats_.avg_processing_time = std::chrono::milliseconds(
                        (stats_.avg_processing_time.count() + processing_time.count()) / 2);
                }
                
            } catch (const std::exception& e) {
                // Log error but continue processing
                // TODO: Add proper logging
                std::cerr << "[" << name_ << "] Processing error: " << e.what() << std::endl;
                
                // Brief sleep to avoid tight error loops
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }
    


private:
    std::string name_;
    std::atomic<bool> running_;
    std::thread thread_;
    
    mutable std::mutex stats_mutex_;
    Stats stats_;
    
    mutable std::mutex interrupt_mutex_;
    std::atomic<bool> interrupt_requested_{false};
};

} // namespace async_pipeline
