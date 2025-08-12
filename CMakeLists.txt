cmake_minimum_required(VERSION 3.16)
project(local-llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find nlohmann/json
find_package(nlohmann_json 3.2.0 QUIET)
if(NOT nlohmann_json_FOUND)
    # If not found, try to find it as a header-only library
    find_path(NLOHMANN_JSON_INCLUDE_DIR nlohmann/json.hpp
        PATHS
        /usr/include
        /usr/local/include
        /opt/homebrew/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
    )
    if(NLOHMANN_JSON_INCLUDE_DIR)
        set(nlohmann_json_INCLUDE_DIRS ${NLOHMANN_JSON_INCLUDE_DIR})
        set(nlohmann_json_FOUND TRUE)
    endif()
endif()

if(NOT nlohmann_json_FOUND)
    message(WARNING "nlohmann/json not found. Configuration file loading will be disabled.")
    add_compile_definitions(DISABLE_JSON_CONFIG)
endif()

# set(CMAKE_BUILD_TYPE Debug)

# stt, tts, and llm options
option(USE_WHISPER   "Enable whisper.cpp for STT"   ON)
option(USE_RKLLM     "Enable rkllm for LLM"         OFF)
option(USE_LLAMA     "Enable llama.cpp for LLM"     ON)
option(USE_Piper     "Enable Piper for TTS"         ON)

find_package(SDL2 REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SOURCES
    src/main.cpp
    src/common-sdl.cpp
    src/common.cpp
    src/server.cpp
)

if(USE_WHISPER)
    add_compile_definitions(STT_BACKEND=WhisperSTT)
    add_definitions(-DUSE_WHISPER)
    list(APPEND SOURCES src/stt_whisper.cpp)
    add_subdirectory(third_party/whisper.cpp ${CMAKE_BINARY_DIR}/whisper-build EXCLUDE_FROM_ALL)
    list(APPEND EXTRA_LIBS whisper)
else()
    message(FATAL_ERROR "No STT backend selected")
endif()

if(USE_LLAMA)
    add_compile_definitions(LLM_BACKEND=LlamaLLM)
    add_definitions(-DUSE_LLAMA)
    list(APPEND SOURCES src/llm_llama.cpp)
    add_subdirectory(third_party/llama.cpp ${CMAKE_BINARY_DIR}/llama-build EXCLUDE_FROM_ALL)
    list(APPEND EXTRA_LIBS llama)
else()
    message(FATAL_ERROR "No LLM backend selected")
endif()

if(USE_Piper)
    add_compile_definitions(TTS_BACKEND=TTSPiper)
    add_definitions(-DUSE_Piper)
    list(APPEND SOURCES src/tts_piper.cpp)
else()
    message(FATAL_ERROR "No TTS backend selected")
endif()

add_executable(local-llm ${SOURCES})
target_include_directories(local-llm PRIVATE ${SDL2_INCLUDE_DIRS} ${nlohmann_json_INCLUDE_DIRS})
target_link_libraries(local-llm PRIVATE whisper llama ${SDL2_LIBRARIES} pthread)
