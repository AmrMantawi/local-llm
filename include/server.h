#pragma once

#include <string>
#include <atomic>

class ILLM;

int run_server(const std::string &socketPath, ILLM &llm, std::atomic<bool> &keepRunning);


