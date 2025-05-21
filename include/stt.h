#pragma once
#include <string>
#include <vector>

class ISTT {
public:
  /// Initialize STT.
  /// @param modelPath path to STT model.
  /// @return true on success, false on failure.
  virtual bool init(const std::string &modelPath) = 0;

  /// Transcribe a single audio buffer.
  /// @param pcmf32 input audio.
  /// @param out resulting UTF-8 text.
  /// @return true on success, false on failure.
  virtual bool transcribe(const std::vector<float> &pcmf32, std::string &outText) = 0;

  /// Release any resources held by STT.
  virtual void shutdown() = 0;
};

