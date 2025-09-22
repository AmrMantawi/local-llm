// src/main.cpp

#include "config_manager.h"
#include "pipeline_manager.h"
#include "async_pipeline_factory.h"

#include <SDL2/SDL.h>
#include <iostream>
#include <vector>
#include <string>
#include <csignal>
#include <atomic>
#include <thread>
#include <chrono>
#include "pipeline_manager.h"

// Server functionality
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cerrno>
#include <cstring>
#include <nlohmann/json.hpp>


// Graceful shutdown on Ctrl+C
static std::atomic<bool> keep_running{true};

static void handle_sigint(int) 
{
    keep_running = false;
}

// Forward declarations
int run_cli_mode(std::atomic<bool>& keep_running);
int run_server_mode(const std::string& socketPath, std::atomic<bool>& keep_running);

int main(int argc, char** argv) {

    // CLI flags: --config /path/models.json, --socket /tmp/local-llm.sock, --server
    const char* configPath = "/usr/share/local-llm/config/models.json";
    std::string socketPath = "/run/local-llm.sock";
    bool server_mode = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if ((arg == "--config" || arg == "-c") && i + 1 < argc) {
            configPath = argv[++i];
        } else if ((arg == "--socket" || arg == "-s") && i + 1 < argc) {
            socketPath = argv[++i];
        } else if (arg == "--server") {
            server_mode = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: local-llm [--server] [--config /path/models.json] [--socket /run/local-llm.sock]\n";
            std::cout << "  --server   Run in server mode (default: CLI mode)\n";
            return 0;
        }
    }

    // Load configuration
    auto& config = ConfigManager::getInstance();
    if (!config.loadConfig(configPath)) {
        std::cout << "Using default configuration (config file not found or invalid)" << std::endl;
    }

    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);

    if (server_mode) {
        // Server mode: use async pipeline with TEXT_ONLY mode
        return run_server_mode(socketPath, keep_running);
    } else {
        // CLI mode: use async pipeline with VOICE_ASSISTANT mode
        return run_cli_mode(keep_running);
    }
}

// Pipeline implementation for CLI mode
int run_cli_mode(std::atomic<bool>& keep_running) {
    std::cout << "Starting pipeline for CLI mode...\n";
    
    try {
        // Create voice assistant pipeline (full Audio → STT → LLM → TTS chain)
        auto pipeline = async_pipeline::PipelineFactory::create_pipeline(async_pipeline::PipelineMode::VOICE_ASSISTANT);
        if (!pipeline) {
            std::cerr << "Failed to create pipeline\n";
            return 1;
        }
        
        // Set interrupt flag for queue operations
        pipeline->set_interrupt_flag(&keep_running);
        
        // Start the pipeline
        if (!pipeline->start()) {
            std::cerr << "Failed to start pipeline\n";
            return 1;
        }
        
        std::cout << "Pipeline started. Listening for speech... (press Ctrl+C to stop)\n";
        std::cout << "Pipeline components running in parallel threads:\n";
        std::cout << "  - Audio capture with VAD\n";
        std::cout << "  - Speech-to-text transcription\n"; 
        std::cout << "  - Language model generation\n";
        std::cout << "  - Text-to-speech synthesis\n\n";
        
        // Main loop - wait for pipeline to complete
        while (keep_running && pipeline->is_running()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
        
        std::cout << "\nStopping pipeline...\n";
        pipeline->stop();
        
        std::cout << "Pipeline stopped successfully\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Pipeline error: " << e.what() << std::endl;
        return 1;
    }
}

// Server functionality using async pipeline
namespace {
int create_and_listen(const std::string &socketPath) {
    int fd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (fd < 0) {
        std::perror("socket");
        return -1;
    }

    ::unlink(socketPath.c_str());

    sockaddr_un addr{};
    addr.sun_family = AF_UNIX;
    std::snprintf(addr.sun_path, sizeof(addr.sun_path), "%s", socketPath.c_str());

    if (::bind(fd, reinterpret_cast<sockaddr *>(&addr), sizeof(addr)) < 0) {
        std::perror("bind");
        ::close(fd);
        return -1;
    }

    if (::listen(fd, 128) < 0) {
        std::perror("listen");
        ::close(fd);
        ::unlink(socketPath.c_str());
        return -1;
    }

    // socket permissions (optional; systemd socket units usually handle perms)
    ::chmod(socketPath.c_str(), 0660);
    return fd;
}

void handle_client_with_pipeline(int client_fd, async_pipeline::PipelineManager& pipeline) {
    FILE *fp = fdopen(client_fd, "r+");
    if (!fp) {
        ::close(client_fd);
        return;
    }
    
    char *line = nullptr;
    size_t len = 0;
    ssize_t nread = getline(&line, &len, fp);
    if (nread <= 0) {
        fclose(fp);
        free(line);
        return;
    }

    // Parse JSON request
    std::string response;
    try {
        std::string firstLine(line, static_cast<size_t>(nread));
        auto req = nlohmann::json::parse(firstLine);
        std::string prompt = req.value("prompt", "");
        
        if (prompt.empty()) {
            throw std::runtime_error("missing prompt");
        }
        
        // Use pipeline's text processing instead of direct LLM call
        if (!pipeline.process_text_input(prompt, response)) {
            throw std::runtime_error("pipeline processing failed");
        }
        
        nlohmann::json resp{{"response", response}};
        std::string out = resp.dump();
        out.push_back('\n');
        fwrite(out.data(), 1, out.size(), fp);
        fflush(fp);
        
    } catch (const std::exception &e) {
        nlohmann::json err{{"error", e.what()}};
        std::string out = err.dump();
        out.push_back('\n');
        fwrite(out.data(), 1, out.size(), fp);
        fflush(fp);
    }
    
    fclose(fp);
    free(line);
}
} // namespace

// Server mode implementation using async pipeline
int run_server_mode(const std::string& socketPath, std::atomic<bool>& keep_running) {
    std::cout << "Starting server mode with async pipeline...\n";
    
    try {
        // Create voice assistant pipeline (full Audio → STT → LLM → TTS chain) with alternate text input option (Text → LLM → Text)
        auto pipeline = async_pipeline::PipelineFactory::create_pipeline(async_pipeline::PipelineMode::VOICE_ASSISTANT_WITH_ALT_TEXT);
        if (!pipeline) {
            std::cerr << "Failed to create pipeline\n";
            return 1;
        }
        
        // Set interrupt flag for queue operations
        pipeline->set_interrupt_flag(&keep_running);
        
        // Start the pipeline
        if (!pipeline->start()) {
            std::cerr << "Failed to start pipeline\n";
            return 1;
        }
        
        std::cout << "Pipeline started with voice assistant + alt text mode\n";
        
        // Create server socket
        int listen_fd = create_and_listen(socketPath);
        if (listen_fd < 0) {
            pipeline->stop();
            return 1;
        }
        
        std::cout << "Server listening on " << socketPath << std::endl;
        std::cout << "Send JSON requests: {\"prompt\": \"your text here\"}\n\n";
        
        std::vector<std::thread> workers;
        while (keep_running && pipeline->is_running()) {
            int client_fd = ::accept(listen_fd, nullptr, nullptr);
            if (client_fd < 0) {
                if (errno == EINTR) continue;
                std::perror("accept");
                break;
            }
            
            // Handle each client in a separate thread using the pipeline
            workers.emplace_back([client_fd, &pipeline]() mutable {
                handle_client_with_pipeline(client_fd, *pipeline);
            });
            // detach to avoid accumulating join() responsibilities
            workers.back().detach();
        }
        
        ::close(listen_fd);
        ::unlink(socketPath.c_str());
        
        // Stop pipeline
        pipeline->stop();
        
        std::cout << "Server stopped.\n";
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Server error: " << e.what() << std::endl;
        return 1;
    }
}