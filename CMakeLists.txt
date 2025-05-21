cmake_minimum_required(VERSION 3.16)
project(local-llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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


add_executable(local-llm ${SOURCES})
target_include_directories(local-llm PRIVATE ${SDL2_INCLUDE_DIRS})
target_link_libraries(local-llm PRIVATE whisper ${SDL2_LIBRARIES} pthread)
