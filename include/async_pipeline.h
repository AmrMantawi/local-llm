#pragma once

#include <vector>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <thread>
#include <atomic>
#include <chrono>
#include <iostream>

namespace async_pipeline {

// Forward declarations
class STTProcessor;
class LLMProcessor;
class TTSProcessor;
class PipelineManager;

enum class PopResult {
    SUCCESS,        // Item successfully popped
    EMPTY,          // Queue is empty
    SHUTDOWN,       // Queue is shutting down
    INTERRUPTED,    // External interrupt requested
    TIMEOUT         // Timeout exceeded
};

#ifdef ENABLE_STATS_LOGGING
struct MessageStats {
    std::chrono::steady_clock::time_point timestamp;
    
    MessageStats() {
        timestamp = std::chrono::steady_clock::now();
    }
    
    // Helper methods
    std::chrono::milliseconds age() const {
        auto now = std::chrono::steady_clock::now();
        return std::chrono::duration_cast<std::chrono::milliseconds>(now - timestamp);
    }
};
#endif

struct TextMessage {
    std::string text;
    
#ifdef ENABLE_STATS_LOGGING
    MessageStats stats;
    
    // Convenience methods for stats access
    std::chrono::milliseconds age() const { return stats.age(); }
#endif
    
    TextMessage() {
#ifdef ENABLE_STATS_LOGGING
        stats = MessageStats();
#endif
    }
    TextMessage(std::string txt)
        : text(std::move(txt)) {
#ifdef ENABLE_STATS_LOGGING
        stats = MessageStats();
#endif
    }
};

struct AudioChunkMessage {
    std::vector<int16_t> audio_data;
    unsigned int sample_rate;
    
#ifdef ENABLE_STATS_LOGGING
    MessageStats stats;
    
    // Convenience methods for stats access
    std::chrono::milliseconds age() const { return stats.age(); }
#endif
    
    AudioChunkMessage() : sample_rate(22050) {
#ifdef ENABLE_STATS_LOGGING
        stats = MessageStats();
#endif
    }
    AudioChunkMessage(std::vector<int16_t> audio, unsigned int rate = 22050)
        : audio_data(std::move(audio)), sample_rate(rate) {
#ifdef ENABLE_STATS_LOGGING
        stats = MessageStats();
#endif
    }
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

#ifdef ENABLE_STATS_LOGGING
    MessageStats stats;
    
    // Convenience methods for stats access
    std::chrono::milliseconds age() const { return stats.age(); }
#endif
    
    ControlMessage(Type t) 
        : type(t) {
#ifdef ENABLE_STATS_LOGGING
        stats = MessageStats();
#endif
    }
};

/**
 * Thread-safe queue implementation for inter-component communication
 */
template<typename T>
class SafeQueue {
public:
    explicit SafeQueue(size_t max_size = 100, std::atomic<bool>* interrupt_flag = nullptr) 
        : max_size_(max_size), shutdown_(false), interrupt_flag_(interrupt_flag) {}
    
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

    // Pop with timeout - returns PopResult indicating success or failure reason
    PopResult pop(T& item, std::chrono::milliseconds timeout = std::chrono::milliseconds(100)) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        if (!not_empty_.wait_for(lock, timeout, [this] { 
            return !queue_.empty() || shutdown_ || external_interrupt_requested(); 
        })) {
            return PopResult::TIMEOUT;
        }
        
        if (shutdown_) return PopResult::SHUTDOWN;
        if (external_interrupt_requested()) return PopResult::INTERRUPTED;
        if (queue_.empty()) return PopResult::EMPTY;
        
        item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return PopResult::SUCCESS;
    }
    
    // Blocking pop - waits indefinitely until item available or shutdown
    PopResult pop_blocking(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        
        not_empty_.wait(lock, [this] { 
            return !queue_.empty() || shutdown_ || external_interrupt_requested(); 
        });
        
        if (shutdown_) return PopResult::SHUTDOWN;
        if (external_interrupt_requested()) return PopResult::INTERRUPTED;
        if (queue_.empty()) return PopResult::EMPTY;
        
        item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return PopResult::SUCCESS;
    }
    
    // Non-blocking pop
    PopResult try_pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (shutdown_) return PopResult::SHUTDOWN;
        if (external_interrupt_requested()) return PopResult::INTERRUPTED;
        if (queue_.empty()) return PopResult::EMPTY;
        
        item = std::move(queue_.front());
        queue_.pop();
        not_full_.notify_one();
        return PopResult::SUCCESS;
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
    std::atomic<bool>* interrupt_flag_;
    
    bool external_interrupt_requested() const {
        return interrupt_flag_ && interrupt_flag_->load(std::memory_order_acquire);
    }
};

/**
 * Base processor class with common thread management and signal-based control
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
        
        // Signal any waiting threads to wake up
        signal_control(ControlMessage(ControlMessage::SHUTDOWN));
        
        if (thread_.joinable()) {
            thread_.join();
        }
        cleanup();
    }
    
    bool is_running() const { return running_; }
    const std::string& name() const { return name_; }
    
    // Signal-based control system
    void signal_control(const ControlMessage& msg) {
        {
            std::lock_guard<std::mutex> lock(control_mutex_);
            control_queue_.push(msg);
        }
        control_signal_.notify_one(); // Wake up processor immediately
        // Reduce logging noise: log only shutdown and pause/resume at info level
        if (msg.type == ControlMessage::SHUTDOWN || msg.type == ControlMessage::PAUSE || msg.type == ControlMessage::RESUME) {
            std::cout << "[" << name_ << "] Control signal received: " << control_type_to_string(msg.type) << std::endl;
        }
    }
    
    // Check for immediate control signals (non-blocking)
    bool check_control_signal(ControlMessage& msg) {
        std::lock_guard<std::mutex> lock(control_mutex_);
        if (!control_queue_.empty()) {
            msg = control_queue_.front();
            control_queue_.pop();
            return true;
        }
        return false;
    }
    
    // Wait for control signal with timeout (interruptible sleep)
    bool wait_for_control_or_timeout(ControlMessage& msg, std::chrono::milliseconds timeout) {
        std::unique_lock<std::mutex> lock(control_mutex_);
        if (control_signal_.wait_for(lock, timeout, [this] { return !control_queue_.empty() || !running_; })) {
            if (!control_queue_.empty()) {
                msg = control_queue_.front();
                control_queue_.pop();
                return true;
            }
        }
        return false;
    }
    
    // Legacy interruption support (deprecated - use signal_control instead)
    virtual void interrupt() {
        signal_control(ControlMessage(ControlMessage::INTERRUPT));
    }
    
    bool is_interrupt_requested() const {
        std::lock_guard<std::mutex> lock(control_mutex_);
        return !control_queue_.empty() && control_queue_.front().type == ControlMessage::INTERRUPT;
    }
    
#ifdef ENABLE_STATS_LOGGING
    // Get processing statistics
    struct Stats {
        uint64_t messages_processed = 0;
        std::chrono::milliseconds avg_processing_time{0};
        uint64_t control_signals_received = 0;
        std::chrono::milliseconds avg_control_response_time{0};
        
        void print() const {
            std::cout << "Processed: " << messages_processed
                      << ", Avg Time: " << avg_processing_time.count() << "ms"
                      << ", Control signals: " << control_signals_received
                      << ", Avg control response: " << avg_control_response_time.count() << "ms"
                      << std::endl;
        }
    };
    
    Stats get_stats() const {
        std::lock_guard<std::mutex> lock(stats_mutex_);
        return stats_;
    }
#endif

protected:
    // Override in derived classes
    virtual bool initialize() = 0;
    virtual void process() = 0;
    virtual void cleanup() {}
    
    
    // Override to handle specific control messages immediately
    virtual bool handle_control_message(const ControlMessage& /*msg*/) {
        return false; // Return true if handled, false to continue with default processing
    }
    
    // Main thread loop with signal-based control
    void run() {
        while (running_) {
            try {
                process();
                
                // Check for any pending control messages after processing
                ControlMessage control_msg(ControlMessage::INTERRUPT);
                while (check_control_signal(control_msg)) {
                    // Handle control message
                    bool handled = handle_control_message(control_msg);
                    if (!handled) {
                        // Default control handling
                        if (control_msg.type == ControlMessage::SHUTDOWN) {
# ifdef ENABLE_STATS_LOGGING
                            auto n = stats_.control_signals_received++;
                            stats_.avg_control_response_time = std::chrono::milliseconds(
                                (stats_.avg_control_response_time.count() * (n - 1) + control_msg.age().count()) / n
                            );
#endif
                            return; // Exit the run loop
                        }
                    }
                }                
            } catch (const std::exception& e) {
                // Log error but continue processing
                std::cerr << "[" << name_ << "] Processing error: " << e.what() << std::endl;
                
                // Brief sleep to avoid tight error loops
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
            }
        }
    }
    
    // Convert control message type to string for logging
    std::string control_type_to_string(ControlMessage::Type type) const {
        switch (type) {
            case ControlMessage::INTERRUPT: return "INTERRUPT";
            case ControlMessage::FLUSH_QUEUES: return "FLUSH_QUEUES";
            case ControlMessage::PAUSE: return "PAUSE";
            case ControlMessage::RESUME: return "RESUME";
            case ControlMessage::SHUTDOWN: return "SHUTDOWN";
            default: return "UNKNOWN";
        }
    }
    


protected:
#ifdef ENABLE_STATS_LOGGING
    mutable std::mutex stats_mutex_;
    Stats stats_;
#endif

private:
    std::string name_;
    std::atomic<bool> running_;
    std::thread thread_;
    
    // Signal-based control system
    mutable std::mutex control_mutex_;
    std::condition_variable control_signal_;
    std::queue<ControlMessage> control_queue_;
};

} // namespace async_pipeline
