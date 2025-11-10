#include "async_processors.h"
#include "config_manager.h"
#include "common.h"
#include <iostream>

namespace async_pipeline {

// Helper functions
inline void fade_and_trim_tail_ms(AudioChunkMessage& m, double fade_ms, double fade_strength, int channels = 1) {
    if (fade_ms <= 0 || m.sample_rate == 0 || channels <= 0) return;

    const size_t per_ch = static_cast<size_t>(std::llround(fade_ms * m.sample_rate / 1000.0));
    const size_t total  = per_ch * static_cast<size_t>(channels);
    if (total == 0 || total >= m.audio_data.size()) { 
        m.audio_data.clear(); 
        return; 
    }

    // Map fade_strength [0–100] → curve exponent [1–5]
    float exponent = 1.0f + (fade_strength / 25.0f); // e.g. 0→1.0, 100→5.0
    const size_t start = m.audio_data.size() - total;

    for (size_t i = 0; i < total; ++i) {
        float t = static_cast<float>(i + 1) / static_cast<float>(total);
        float g = std::pow(1.0f - t, exponent);
        int32_t v = static_cast<int32_t>(std::lround(m.audio_data[start + i] * g));
        m.audio_data[start + i] = static_cast<int16_t>(std::clamp(v, -32768, 32767));
    }

    m.audio_data.resize(start); // trim off faded tail
}

// STTProcessor implementation
STTProcessor::STTProcessor(SafeQueue<TextMessage>& output_queue, std::unique_ptr<ISTT> stt_backend)
    : BaseProcessor("STTProcessor"), output_queue_(output_queue),
      stt_(std::move(stt_backend)), audio_(nullptr),
      is_in_speech_sequence_(false) {
}

bool STTProcessor::initialize() {
    if (!stt_) {
        std::cerr << "[STTProcessor] No STT backend provided" << std::endl;
        return false;
    }
    
    auto& config = ConfigManager::getInstance();
    const std::string whisper_model_path = config.getSTTModelPath();
    
    if (!stt_->init(whisper_model_path)) {
        std::cerr << "[STTProcessor] Failed to initialize STT backend" << std::endl;
        return false;
    }
    
    // Cache config values once during initialization
    sample_rate_ = config.getAudioSampleRate();
    buffer_ms_ = config.getAudioBufferMs();
    vad_threshold_ = config.getVadThreshold();
    vad_capture_ms_ = config.getVadCaptureMs();
    
    // Initialize audio capture
    audio_ = new audio_async(buffer_ms_);
    bool audio_initialized = false;
    for (int attempt = 1; attempt <= 8; ++attempt) {
        if (audio_->init(-1, sample_rate_)) {
            audio_initialized = true;
            break;
        }
        std::cerr << "[STTProcessor] Audio init attempt " << attempt << " failed, retrying in 500ms..." << std::endl;
        std::this_thread::sleep_for(std::chrono::milliseconds(500));
    }
    
    if (!audio_initialized) {
        std::cerr << "[STTProcessor] Failed to initialize audio capture after 8 attempts" << std::endl;
        return false;
    }
    
    if (!audio_->resume()) {
        std::cerr << "[STTProcessor] Failed to start audio capture" << std::endl;
        return false;
    }
    
    std::cout << "[STTProcessor] Initialized successfully" << std::endl;
    return true;
}

bool STTProcessor::handle_control_message(const ControlMessage& msg) {
    if (msg.type == ControlMessage::INTERRUPT || 
        msg.type == ControlMessage::FLUSH_QUEUES) {
        // Flush our output queue
        size_t flushed = output_queue_.flush();
        if (flushed > 0) {
            std::cout << "[STTProcessor] Flushed " << flushed << " pending text messages" << std::endl;
        }
        return true; // Handled
    }
    return false; // Not handled, use default behavior
}

void STTProcessor::process() {
    // Check if we should stop processing
    if (!is_running()) {
        return;
    }
    
    if (!audio_) {
        std::cerr << "[STTProcessor] WARNING: audio_ is null!" << std::endl;
        // Use interruptible sleep instead of fixed sleep
        ControlMessage control_msg(ControlMessage::INTERRUPT);
        wait_for_control_or_timeout(control_msg, std::chrono::milliseconds(100));
        return;
    }
    
    std::vector<float> audio;
    
    // Get recent audio for VAD analysis with interruptible sleep
    audio_->get(vad_pre_window_ms_, audio);
    
    if (audio.empty()) {
        // No audio available, use interruptible sleep
        ControlMessage control_msg(ControlMessage::INTERRUPT);
        wait_for_control_or_timeout(control_msg, std::chrono::milliseconds(50));
        return;
    }
    
    // Use cached config values (no repeated config access!)
    bool voice_detected = ::vad_simple(
        audio, 
        sample_rate_, 
        1250, // vad_start_ms
        vad_threshold_, 
        100.0f, // vad_freq_cutoff
        false // vad_print_energy
    );
    
    if (voice_detected) {
        // Capture full audio window
        audio_->get(vad_capture_ms_, audio);
        
        if (!audio.empty()) {
            AudioChunkMessage audio_msg; // For stats
            
            // Then transcribe the audio
            std::string transcribed_text;
            bool success = stt_->transcribe(audio, transcribed_text);
            
            if (success && !transcribed_text.empty()) {
                // Create text message
                TextMessage text_msg(transcribed_text);
                
                // Push to output queue with blocking push to handle queue full situations
                if (!output_queue_.push_blocking(std::move(text_msg))) {
                    // Queue was shut down, stop processing
                    return;
                } else {
#ifdef ENABLE_STATS_LOGGING
                    auto n = stats_.messages_processed++;
                    auto msg_age = audio_msg.age().count();
                    auto current_avg = stats_.avg_processing_time.count();
                    if (msg_age >= 0 && n > 0) {
                        stats_.avg_processing_time = std::chrono::milliseconds(
                            (current_avg * (n - 1) + msg_age) / n
                        );
                    }
#endif
                    std::cout << "[STTProcessor] → " << transcribed_text << std::endl;
                }
            }
            
            // Clear audio buffer
            audio_->clear();
        }
    }
}

void STTProcessor::cleanup() {
    if (audio_) {
        audio_->pause();
        delete audio_;
        audio_ = nullptr;
    }
    if (stt_) {
        stt_->shutdown();
    }
    std::cout << "[STTProcessor] Cleanup completed" << std::endl;
}

// LLMProcessor implementation
LLMProcessor::LLMProcessor(SafeQueue<TextMessage>& input_queue, SafeQueue<TextMessage>& output_queue,
                           std::unique_ptr<ILLM> llm_backend,
                           SafeQueue<TextMessage>* alt_input_queue,
                           SafeQueue<TextMessage>* alt_output_queue)
    : BaseProcessor("LLMProcessor"), input_queue_(input_queue), output_queue_(output_queue),
      alt_input_queue_(alt_input_queue), alt_output_queue_(alt_output_queue),
      llm_(std::move(llm_backend)) {
}

bool LLMProcessor::initialize() {
    if (!llm_) {
        std::cerr << "[LLMProcessor] No LLM backend provided" << std::endl;
        return false;
    }
    
    // Initialize LLM backend (consistent with TTSProcessor pattern)
    auto& config = ConfigManager::getInstance();
    const std::string llm_model_path = config.getLLMModelPath();
    
    if (!llm_->init(llm_model_path)) {
        std::cerr << "[LLMProcessor] Failed to initialize LLM backend" << std::endl;
        return false;
    }
    
    std::cout << "[LLMProcessor] Initialized successfully" << std::endl;
    return true;
}

bool LLMProcessor::handle_control_message(const ControlMessage& msg) {
    if (msg.type == ControlMessage::INTERRUPT || 
        msg.type == ControlMessage::FLUSH_QUEUES) {
        // Flush input and output queues
        size_t input_flushed = input_queue_.flush();
        size_t output_flushed = output_queue_.flush();
        
        // Flush alt queues if they exist
        size_t alt_input_flushed = 0;
        size_t alt_output_flushed = 0;
        if (alt_input_queue_) {
            alt_input_flushed = alt_input_queue_->flush();
        }
        if (alt_output_queue_) {
            alt_output_flushed = alt_output_queue_->flush();
        }
        
        if (input_flushed > 0 || output_flushed > 0 || alt_input_flushed > 0 || alt_output_flushed > 0) {
            std::cout << "[LLMProcessor] Flushed " << input_flushed << " input, " 
                      << output_flushed << " output, " << alt_input_flushed << " alt input, "
                      << alt_output_flushed << " alt output messages" << std::endl;
        }
        return true; // Handled
    }
    return false; // Not handled, use default behavior
}

void LLMProcessor::process() {
    // Check if we should stop processing
    if (!is_running()) {
        return;
    }
    
    TextMessage input_msg;
    PopResult result = input_queue_.pop_blocking(input_msg);
    
    if (result == PopResult::SUCCESS) {
        std::cout << "[LLMProcessor] Processing: " << input_msg.text << std::endl;
        
        // Generate response
        std::string response;
        bool success = llm_->generate_async(input_msg.text, response, 
            [this, &input_msg](const std::string& text_chunk) {
                // Create response message
                TextMessage response_msg(text_chunk);
                
                // Push to output queue with blocking push
                if (!output_queue_.push_blocking(std::move(response_msg))) {
                    // Queue was shut down, stop processing
                    return;
                } else {
#ifdef ENABLE_STATS_LOGGING
                    auto n = stats_.messages_processed++;
                    auto msg_age = input_msg.age().count();
                    auto current_avg = stats_.avg_processing_time.count();
                    if (msg_age >= 0 && n > 0) {
                        stats_.avg_processing_time = std::chrono::milliseconds(
                            (current_avg * (n - 1) + msg_age) / n
                        );
                    }
#endif
                    std::cout << "[LLMProcessor] → " << text_chunk << std::endl;
                }
            });
        
        if (!success) {
            std::cerr << "[LLMProcessor] Failed to generate response for: " << input_msg.text << std::endl;
        }
    } else if (result == PopResult::SHUTDOWN) {
        // Queue is shutting down, stop processing
        return;
    } else if (result == PopResult::INTERRUPTED) {
        // External interrupt requested, continue to check control messages
        return;
    }
}

void LLMProcessor::cleanup() {
    if (llm_) {
        llm_->shutdown();
    }
    std::cout << "[LLMProcessor] Cleanup completed" << std::endl;
}

// TTSProcessor implementation
TTSProcessor::TTSProcessor(SafeQueue<TextMessage>& input_queue, std::unique_ptr<ITTS> tts_backend, std::atomic<bool>* interrupt_flag)
    : BaseProcessor("TTSProcessor"), input_queue_(input_queue),
      tts_(std::move(tts_backend)), is_speaking_(false), tts_pid_(-1), interrupt_flag_(interrupt_flag) {
}

void TTSProcessor::stop() {
    if (!is_running()) return;
    
    // Now call the base stop() method to join the thread
    BaseProcessor::stop();
}

bool TTSProcessor::initialize() {
    if (!tts_) {
        std::cerr << "[TTSProcessor] No TTS backend provided" << std::endl;
        return false;
    }
    
    // Create audio output queue using the same interrupt flag
    audio_output_queue_ = std::make_unique<SafeQueue<AudioChunkMessage>>(50, interrupt_flag_);
    
    if (!tts_->init()) {
        std::cerr << "[TTSProcessor] Failed to initialize TTS backend" << std::endl;
        return false;
    }
            
    // Initialize AudioOutputProcessor with the queue
    audio_output_processor_ = std::make_unique<AudioOutputProcessor>(*audio_output_queue_);
    if (!audio_output_processor_->start()) {
        std::cerr << "[TTSProcessor] Failed to start AudioOutputProcessor" << std::endl;
        return false;
    }

    // Setup Unix socket for face control
    if (!setup_unix_socket()) {
        std::cerr << "[TTSProcessor] Failed to setup Unix socket" << std::endl;
        return false;
    }
    
    // Setup shared memory for phoneme data
    if (!setup_shared_memory()) {
        std::cerr << "[TTSProcessor] Failed to setup shared memory" << std::endl;
        return false;
    }
    
    face_shown_ = false;
    
    std::cout << "[TTSProcessor] Initialized successfully with audio output processor" << std::endl;
    return true;
}

bool TTSProcessor::handle_control_message(const ControlMessage& msg) {
    if (msg.type == ControlMessage::INTERRUPT || 
        msg.type == ControlMessage::FLUSH_QUEUES) {
        // Flush input queue
        size_t flushed = input_queue_.flush();
        if (flushed > 0) {
            std::cout << "[TTSProcessor] Interrupted! Flushed " << flushed << " pending TTS messages" << std::endl;
        }
        // Stop current speech gracefully
        interrupt_current_speech();
        std::cout << "[TTSProcessor] Interrupt handled, ready for new speech" << std::endl;
        return true; // Handled
    }
    return false; // Not handled, use default behavior
}

void TTSProcessor::process() {
    // Check if we should stop processing
    if (!is_running()) {
        return;
    }
    
    TextMessage text_msg;
    PopResult result = input_queue_.pop_blocking(text_msg);
    
    if (result == PopResult::SUCCESS) {
        // Speak the chunk and get audio data
        std::cout << "[TTSProcessor] Speaking: " << text_msg.text << std::endl;
        
        // Create AudioChunkMessage to receive the audio data
        AudioChunkMessage audio_chunk;
        bool success = false;
        if(face_shown_) {
            std::vector<PhonemeTimingInfo> phoneme_timings;
            success = tts_->speakWithPhonemeTimings(text_msg.text, audio_chunk, phoneme_timings);
            
            // Send phoneme data to shared memory if available
            if (success && !phoneme_timings.empty()) {
                send_phoneme_data(phoneme_timings);
            }
        } else {
            success = tts_->speak(text_msg.text, audio_chunk);
        }
        
        if (success && !audio_chunk.audio_data.empty()) {
            fade_and_trim_tail_ms(audio_chunk, 325, 120);
            // Queue the audio chunk for playback with blocking push
            if (!audio_output_queue_->push_blocking(std::move(audio_chunk))) {
                // Queue was shut down, stop processing
                return;
            } else {
#ifdef ENABLE_STATS_LOGGING
                    auto n = stats_.messages_processed++;
                    auto msg_age = text_msg.age().count();
                    auto current_avg = stats_.avg_processing_time.count();
                    if (msg_age >= 0 && n > 0) {
                        stats_.avg_processing_time = std::chrono::milliseconds(
                            (current_avg * (n - 1) + msg_age) / n
                        );
                    }
#endif
            }
        } else {
            std::cerr << "[TTSProcessor] Failed to speak: " << text_msg.text << std::endl;
        }
    } else if (result == PopResult::SHUTDOWN) {
        // Queue is shutting down, stop processing
        return;
    } else if (result == PopResult::INTERRUPTED) {
        // External interrupt requested, continue to check control messages
        return;
    }
    // For other cases (EMPTY), the loop will continue
}

void TTSProcessor::cleanup() {
    // Cleanup Unix socket
    cleanup_socket();
    
    // Cleanup shared memory
    cleanup_shared_memory();
    
    // Ensure audio playback is interrupted and the output queue is unblocked
    if (audio_output_processor_) {
        // Stop any ongoing ALSA playback and flush pending chunks
        audio_output_processor_->interrupt_audio_immediately();
    }

    if (audio_output_queue_) {
        // Wake any blocking pop() in AudioOutputProcessor
        audio_output_queue_->shutdown();
    }

    // Now stop the internal audio output processor thread cleanly
    if (audio_output_processor_) {
        audio_output_processor_->stop();
        audio_output_processor_.reset();
    }
    // Release the queue after the processor has been stopped
    if (audio_output_queue_) {
        audio_output_queue_.reset();
    }
    
    if (tts_) {
        tts_->shutdown();
    }
    std::cout << "[TTSProcessor] Cleanup completed" << std::endl;
}

void TTSProcessor::interrupt_current_speech() {
    if (audio_output_processor_) {
        std::cout << "[TTSProcessor] Using immediate audio interruption" << std::endl;
        audio_output_processor_->interrupt_audio_immediately();
    }
}

// AudioOutputProcessor implementation
AudioOutputProcessor::AudioOutputProcessor(SafeQueue<AudioChunkMessage>& input_queue)
    : BaseProcessor("AudioOutputProcessor"), input_queue_(input_queue),
      alsa_handle_(nullptr), sample_rate_(22050) {
}

void AudioOutputProcessor::interrupt_audio_immediately() {
    // Clear the audio queue to prevent queued audio from playing
    size_t flushed = input_queue_.flush();
    if (flushed > 0) {
        std::cout << "[AudioOutputProcessor] Flushed " << flushed << " queued audio chunks" << std::endl;
    }
    
    // Stop ALSA playback immediately
    if (alsa_handle_) {
        snd_pcm_drop(alsa_handle_);  // Stop immediately, don't drain buffer
        snd_pcm_prepare(alsa_handle_); // Prepare for next playback
        std::cout << "[AudioOutputProcessor] Stopped ALSA playback immediately" << std::endl;
    }
}

bool AudioOutputProcessor::initialize() {
    std::cout << "[AudioOutputProcessor] Initializing ALSA..." << std::endl;
    return init_audio_device();
}

void AudioOutputProcessor::process() {
    // Check if we should stop processing
    if (!is_running()) {
        return;
    }
    
    AudioChunkMessage audio_msg;
    PopResult result = input_queue_.pop_blocking(audio_msg);
    
    if (result == PopResult::SUCCESS) {
        if (!audio_msg.audio_data.empty()) {
            play_audio_chunk(audio_msg.audio_data);
        }
    } else if (result == PopResult::SHUTDOWN) {
        // Queue is shutting down, stop processing
        return;
    } else if (result == PopResult::INTERRUPTED) {
        // External interrupt requested, continue to check control messages
        return;
    }
}

bool AudioOutputProcessor::handle_control_message(const ControlMessage& msg) {
    if (msg.type == ControlMessage::SHUTDOWN) {
        // Ensure audio device is properly closed before shutdown
        std::cout << "[AudioOutputProcessor] Handling SHUTDOWN signal, closing audio device..." << std::endl;
        close_audio_device();
        std::cout << "[AudioOutputProcessor] Cleanup completed" << std::endl;
        return true; // Handled
    }
    return false; // Not handled, use default behavior
}

void AudioOutputProcessor::cleanup() {
    close_audio_device();
    std::cout << "[AudioOutputProcessor] Cleanup completed" << std::endl;
}

bool AudioOutputProcessor::init_audio_device() {
    int err;
    
    if ((err = snd_pcm_open(&alsa_handle_, "default", SND_PCM_STREAM_PLAYBACK, 0)) < 0) {
        std::cerr << "[AudioOutputProcessor] Cannot open audio device: " << snd_strerror(err) << std::endl;
        return false;
    }
    
    // Set hardware parameters
    snd_pcm_hw_params_t *params;
    snd_pcm_hw_params_alloca(&params);
    snd_pcm_hw_params_any(alsa_handle_, params);
    snd_pcm_hw_params_set_access(alsa_handle_, params, SND_PCM_ACCESS_RW_INTERLEAVED);
    snd_pcm_hw_params_set_format(alsa_handle_, params, SND_PCM_FORMAT_S16_LE);
    snd_pcm_hw_params_set_channels(alsa_handle_, params, 1);
    
    // Set sample rate
    unsigned int rate = sample_rate_;
    if ((err = snd_pcm_hw_params_set_rate_near(alsa_handle_, params, &rate, 0)) < 0) {
        std::cerr << "[AudioOutputProcessor] Cannot set sample rate: " << snd_strerror(err) << std::endl;
        snd_pcm_close(alsa_handle_);
        alsa_handle_ = nullptr;
        return false;
    }
    
    // Set period size for low latency
    snd_pcm_uframes_t period_size = 1024;
    if ((err = snd_pcm_hw_params_set_period_size_near(alsa_handle_, params, &period_size, 0)) < 0) {
        std::cerr << "[AudioOutputProcessor] Cannot set period size: " << snd_strerror(err) << std::endl;
        snd_pcm_close(alsa_handle_);
        alsa_handle_ = nullptr;
        return false;
    }
    
    // Set buffer size
    snd_pcm_uframes_t buffer_size = period_size * 4;
    if ((err = snd_pcm_hw_params_set_buffer_size_near(alsa_handle_, params, &buffer_size)) < 0) {
        std::cerr << "[AudioOutputProcessor] Cannot set buffer size: " << snd_strerror(err) << std::endl;
        snd_pcm_close(alsa_handle_);
        alsa_handle_ = nullptr;
        return false;
    }
    
    if ((err = snd_pcm_hw_params(alsa_handle_, params)) < 0) {
        std::cerr << "[AudioOutputProcessor] Cannot set parameters: " << snd_strerror(err) << std::endl;
        snd_pcm_close(alsa_handle_);
        alsa_handle_ = nullptr;
        return false;
    }
    
    // Prepare the PCM device
    if ((err = snd_pcm_prepare(alsa_handle_)) < 0) {
        std::cerr << "[AudioOutputProcessor] Cannot prepare audio device: " << snd_strerror(err) << std::endl;
        snd_pcm_close(alsa_handle_);
        alsa_handle_ = nullptr;
        return false;
    }
    
    std::cout << "[AudioOutputProcessor] ALSA initialized successfully at " << rate << " Hz" << std::endl;
    return true;
}

void AudioOutputProcessor::close_audio_device() {
    if (alsa_handle_) {
        // Drain any remaining audio data
        snd_pcm_drain(alsa_handle_);
        snd_pcm_close(alsa_handle_);
        alsa_handle_ = nullptr;
    }
}

void AudioOutputProcessor::play_audio_chunk(const std::vector<int16_t>& chunk) {
    if (chunk.empty() || !alsa_handle_) {
        return;
    }
    
    const int16_t* data_ptr = chunk.data();
    size_t frames_total = chunk.size();
    while (frames_total > 0) {
        snd_pcm_sframes_t written = snd_pcm_writei(alsa_handle_, data_ptr, frames_total);
        if (written >= 0) {
            data_ptr += written;
            frames_total -= static_cast<size_t>(written);
            continue;
        }
        if (written == -EPIPE) {
            std::cerr << "[AudioOutputProcessor] ALSA underrun, recovering..." << std::endl;
            snd_pcm_prepare(alsa_handle_);
            continue;
        }
        std::cerr << "[AudioOutputProcessor] ALSA write error: " << snd_strerror(written) << std::endl;
        break;
    }
}

// Unix socket and shared memory implementation for TTSProcessor

bool TTSProcessor::setup_unix_socket() {
    // Create Unix socket
    socket_fd_ = socket(AF_UNIX, SOCK_STREAM, 0);
    if (socket_fd_ < 0) {
        std::cerr << "[TTSProcessor] Failed to create Unix socket: " << strerror(errno) << std::endl;
        return false;
    }
    
    // Remove existing socket file
    unlink(socket_path_.c_str());
    
    // Setup socket address
    struct sockaddr_un addr;
    memset(&addr, 0, sizeof(addr));
    addr.sun_family = AF_UNIX;
    strncpy(addr.sun_path, socket_path_.c_str(), sizeof(addr.sun_path) - 1);
    
    // Bind socket
    if (bind(socket_fd_, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
        std::cerr << "[TTSProcessor] Failed to bind Unix socket: " << strerror(errno) << std::endl;
        close(socket_fd_);
        socket_fd_ = -1;
        return false;
    }
    
    // Listen for connections
    if (listen(socket_fd_, 5) < 0) {
        std::cerr << "[TTSProcessor] Failed to listen on Unix socket: " << strerror(errno) << std::endl;
        close(socket_fd_);
        socket_fd_ = -1;
        return false;
    }
    
    // Start socket server thread
    socket_running_ = true;
    socket_thread_ = std::thread(&TTSProcessor::socket_server_thread, this);
    
    std::cout << "[TTSProcessor] Unix socket setup successfully at " << socket_path_ << std::endl;
    return true;
}

void TTSProcessor::socket_server_thread() {
    while (socket_running_) {
        struct sockaddr_un client_addr;
        socklen_t client_len = sizeof(client_addr);
        
        // Accept connection with timeout
        fd_set read_fds;
        FD_ZERO(&read_fds);
        FD_SET(socket_fd_, &read_fds);
        
        struct timeval timeout;
        timeout.tv_sec = 1;
        timeout.tv_usec = 0;
        
        int result = select(socket_fd_ + 1, &read_fds, nullptr, nullptr, &timeout);
        if (result < 0) {
            if (errno == EINTR) continue;
            std::cerr << "[TTSProcessor] Socket select error: " << strerror(errno) << std::endl;
            break;
        }
        
        if (result == 0) continue; // Timeout
        
        int client_fd = accept(socket_fd_, (struct sockaddr*)&client_addr, &client_len);
        if (client_fd < 0) {
            if (errno == EINTR || errno == EAGAIN) continue;
            std::cerr << "[TTSProcessor] Socket accept error: " << strerror(errno) << std::endl;
            continue;
        }
        
        // Read command from client
        char buffer[256];
        ssize_t bytes_read = read(client_fd, buffer, sizeof(buffer) - 1);
        if (bytes_read > 0) {
            buffer[bytes_read] = '\0';
            std::string command(buffer);
            command.erase(command.find_last_not_of(" \n\r\t") + 1); // Trim whitespace
            handle_socket_command(command);
        }
        
        close(client_fd);
    }
}

void TTSProcessor::handle_socket_command(const std::string& command) {
    if (command == "face_show") {
        face_shown_ = true;
        std::cout << "[TTSProcessor] Face display enabled via socket command" << std::endl;
    } else if (command == "face_hide") {
        face_shown_ = false;
        std::cout << "[TTSProcessor] Face display disabled via socket command" << std::endl;
    } else if (command == "face_toggle") {
        face_shown_ = !face_shown_;
        std::cout << "[TTSProcessor] Face display toggled to: " << (face_shown_ ? "enabled" : "disabled") << std::endl;
    } else {
        std::cout << "[TTSProcessor] Unknown socket command: " << command << std::endl;
    }
}

void TTSProcessor::cleanup_socket() {
    socket_running_ = false;
    
    if (socket_thread_.joinable()) {
        socket_thread_.join();
    }
    
    if (socket_fd_ >= 0) {
        close(socket_fd_);
        socket_fd_ = -1;
    }
    
    unlink(socket_path_.c_str());
}

bool TTSProcessor::setup_shared_memory() {
    // Create shared memory object with proper name format
    std::string shm_name = "/" + shared_mem_path_;
    shared_mem_fd_ = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0666);
    if (shared_mem_fd_ < 0) {
        std::cerr << "[TTSProcessor] Failed to create shared memory: " << strerror(errno) << std::endl;
        return false;
    }
    
    // Set size of shared memory
    size_t mem_size = sizeof(PhonemeSharedQueue);
    if (ftruncate(shared_mem_fd_, mem_size) < 0) {
        std::cerr << "[TTSProcessor] Failed to set shared memory size: " << strerror(errno) << std::endl;
        close(shared_mem_fd_);
        shared_mem_fd_ = -1;
        return false;
    }
    
    // Map shared memory
    shared_queue_ = static_cast<PhonemeSharedQueue*>(
        mmap(nullptr, mem_size, PROT_READ | PROT_WRITE, MAP_SHARED, shared_mem_fd_, 0)
    );
    
    if (shared_queue_ == MAP_FAILED) {
        std::cerr << "[TTSProcessor] Failed to map shared memory: " << strerror(errno) << std::endl;
        close(shared_mem_fd_);
        shared_mem_fd_ = -1;
        return false;
    }
    
    // Initialize shared queue
    new (shared_queue_) PhonemeSharedQueue();
    
    std::cout << "[TTSProcessor] Shared memory setup successfully" << std::endl;
    return true;
}

void TTSProcessor::send_phoneme_data(const std::vector<PhonemeTimingInfo>& phonemes) {
    if (!shared_queue_) return;
    
    auto current_time = std::chrono::duration_cast<std::chrono::microseconds>(
        std::chrono::steady_clock::now().time_since_epoch()
    ).count();
    
    for (const auto& phoneme : phonemes) {
        uint32_t write_idx = shared_queue_->header.write_index.load();
        uint32_t next_idx = (write_idx + 1) % PhonemeQueueHeader::MAX_PHONEMES;
        
        // Check if queue is full (next write would overwrite read position)
        if (next_idx == shared_queue_->header.read_index.load()) {
            std::cerr << "[TTSProcessor] Phoneme queue is full, dropping phoneme" << std::endl;
            break;
        }
        
        // Write phoneme data
        shared_queue_->phonemes[write_idx].phoneme_id = phoneme.phoneme_id;
        shared_queue_->phonemes[write_idx].duration_seconds = phoneme.duration_seconds;
        shared_queue_->phonemes[write_idx].timestamp_us = current_time;
        
        // Update write index
        shared_queue_->header.write_index.store(next_idx);
    }
}

void TTSProcessor::cleanup_shared_memory() {
    if (shared_queue_) {
        size_t mem_size = sizeof(PhonemeSharedQueue);
        munmap(shared_queue_, mem_size);
        shared_queue_ = nullptr;
    }
    
    if (shared_mem_fd_ >= 0) {
        close(shared_mem_fd_);
        shared_mem_fd_ = -1;
    }
    
    std::string shm_name = "/" + shared_mem_path_;
    shm_unlink(shm_name.c_str());
}

} // namespace async_pipeline
