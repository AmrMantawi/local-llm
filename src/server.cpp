#include "server.h"
#include "llm.h"

#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <csignal>
#include <cerrno>
#include <cstring>
#include <iostream>
#include <thread>
#include <chrono>
#include <vector>
#include <nlohmann/json.hpp>

// Streaming mode dependencies (VAD + STT)
#include "common-sdl.h"
#include "common.h"
#include "config_manager.h"

#ifdef USE_WHISPER
#include "stt_whisper.h"
using STT = WhisperSTT;
#endif

#ifdef USE_Piper
#include "tts_piper.h"
using TTS = TTSPiper;
#endif

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

void handle_client(int client_fd, ILLM &llm) {
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

    // Determine mode: "stream" (prefix) or JSON one-shot
    std::string firstLine(line, static_cast<size_t>(nread));
    auto starts_with = [](const std::string &s, const char *pfx) -> bool {
        return s.rfind(pfx, 0) == 0; // prefix match
    };

    if (starts_with(firstLine, "stream")) {
        // Streaming mode: capture mic audio, transcribe, generate response, and write text chunks
        // Initialize audio & STT
        auto &config = ConfigManager::getInstance();

        // Use the same defaults as in main.cpp
        const int    VAD_PRE_WINDOW_MS = 2000;
        const int    VAD_START_MS      = 1250;
        const float  VAD_FREQ_CUTOFF   = 100.0f;
        const bool   VAD_PRINT_ENERGY  = false;

        audio_async audio(config.getAudioBufferMs());
        if (!audio.init(-1, config.getAudioSampleRate())) {
            const char *msg = "{\"error\":\"audio.init() failed\"}\n";
            fwrite(msg, 1, std::strlen(msg), fp);
            fflush(fp);
            fclose(fp);
            free(line);
            return;
        }
        audio.resume();

#ifdef USE_WHISPER
        STT stt;
        if (!stt.init(config.getModelPath("stt", "whisper"))) {
            const char *msg = "{\"error\":\"Failed to initialize STT backend\"}\n";
            fwrite(msg, 1, std::strlen(msg), fp);
            fflush(fp);
            audio.pause();
            fclose(fp);
            free(line);
            return;
        }
#endif

#ifdef USE_Piper
        TTS tts;
        bool tts_initialized = false;
        try {
            tts_initialized = tts.init();
        } catch (...) {
            tts_initialized = false;
        }
#endif

        // Inform client
        {
            const char *msg = "Streaming started. Speak now...\n";
            fwrite(msg, 1, std::strlen(msg), fp);
            fflush(fp);
        }

        std::vector<float> pcmf32;
        while (true) {
            // Small sleep to avoid busy loop
            std::this_thread::sleep_for(std::chrono::milliseconds(100));

            // Peek recent audio for VAD
            audio.get(VAD_PRE_WINDOW_MS, pcmf32);
            if (!pcmf32.empty() && ::vad_simple(pcmf32, config.getAudioSampleRate(), VAD_START_MS, config.getVadThreshold(), VAD_FREQ_CUTOFF, VAD_PRINT_ENERGY)) {
                // Capture full window
                audio.get(config.getVadCaptureMs(), pcmf32);

                std::string text;
#ifdef USE_WHISPER
                bool stt_ok = stt.transcribe(pcmf32, text);
#else
                bool stt_ok = false;
#endif
                if (stt_ok && !text.empty()) {
                    std::string lineOut = std::string("→ ") + text + "\n";
                    if (fwrite(lineOut.data(), 1, lineOut.size(), fp) != lineOut.size()) break;
                    fflush(fp);

                    std::string responseText;
                    if (llm.generate(text, responseText)) {
                        std::string respOut = std::string("BMO: ") + responseText + "\n";
                        if (fwrite(respOut.data(), 1, respOut.size(), fp) != respOut.size()) break;
                        fflush(fp);
#ifdef USE_Piper
                        if (tts_initialized) {
                            (void) tts.speak(responseText);
                        }
#endif
                    } else {
                        const char *err = "{\"error\":\"LLM generation failed\"}\n";
                        if (fwrite(err, 1, std::strlen(err), fp) != (int)std::strlen(err)) break;
                        fflush(fp);
                    }
                }
                audio.clear();
            }

            // Check if client disconnected by writing a heartbeat (no-op flush)
            if (ferror(fp)) {
                break;
            }
        }

        audio.pause();
#ifdef USE_Piper
        if (tts_initialized) {
            tts.shutdown();
        }
#endif
        fclose(fp);
        free(line);
        return;
    }

    // One-shot JSON mode
    std::string responseText;
    try {
        auto req = nlohmann::json::parse(firstLine);
        std::string prompt = req.value("prompt", "");
        if (prompt.empty()) {
            throw std::runtime_error("missing prompt");
        }
        if (!llm.generate(prompt, responseText)) {
            throw std::runtime_error("generation failed");
        }
        nlohmann::json resp{{"response", responseText}};
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

int run_server(const std::string &socketPath, ILLM &llm, std::atomic<bool> &keepRunning) {
    int listen_fd = create_and_listen(socketPath);
    if (listen_fd < 0) {
        return 1;
    }

    std::cout << "Server listening on " << socketPath << std::endl;
    std::vector<std::thread> workers;
    while (keepRunning) {
        int client_fd = ::accept(listen_fd, nullptr, nullptr);
        if (client_fd < 0) {
            if (errno == EINTR) continue;
            std::perror("accept");
            break;
        }
        workers.emplace_back([client_fd, &llm]() mutable {
            handle_client(client_fd, llm);
        });
        // detach to avoid accumulating join() responsibilities
        workers.back().detach();
    }

    ::close(listen_fd);
    ::unlink(socketPath.c_str());
    return 0;
}


