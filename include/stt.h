#pragma once
#include <string>
#include <vector>

class ISTT {
public:
  /// Initialize STT.
  /// Model path is automatically retrieved from ConfigManager.
  /// @return true on success, false on failure.
  virtual bool init() = 0;

  /// Transcribe a single audio buffer.
  /// @param pcmf32 input audio.
  /// @param out resulting UTF-8 text.
  /// @return true on success, false on failure.
  virtual bool transcribe(const std::vector<float> &pcmf32, std::string &outText) = 0;

  /// Release any resources held by STT.
  virtual void shutdown() = 0;
};

