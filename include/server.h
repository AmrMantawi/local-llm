#pragma once

#include <string>
#include <atomic>

/// Run the server listening on the given Unix domain socket path.
/// @param socketPath Path to the Unix domain socket
/// @param keepRunning Atomic flag to control server loop
/// @return 0 on clean exit, non-zero on error
int run_server(const std::string &socketPath, std::atomic<bool> &keepRunning);


