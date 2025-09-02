#pragma once
#include <string>

// Forward declaration
namespace async_pipeline {
    struct AudioChunkMessage;
}

class ITTS {
public:
  /// Initialize TTS.
  /// @return true on success, false on failure.
  virtual bool init() = 0;

  /// Speak the given text and write audio to AudioChunkMessage.
  /// @param text UTF-8 text to speak.
  /// @param audio_chunk AudioChunkMessage to write audio data to.
  /// @return true on success, false on failure.
  virtual bool speak(const std::string &text, async_pipeline::AudioChunkMessage& audio_chunk) = 0;

  // TODO: Add async speak with callback for streaming audio

  /// Release any resources held by TTS.
  virtual void shutdown() = 0;
};

