cmake_minimum_required(VERSION 3.16)
project(local-llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Function to detect OS and architecture for dependency downloads
function(detect_platform)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(PLATFORM_OS "linux" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Unsupported OS: ${CMAKE_SYSTEM_NAME}. Only Linux is supported.")
    endif()
    
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(PLATFORM_ARCH "x64" PARENT_SCOPE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(PLATFORM_ARCH "arm64" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}. Only x86_64 and arm64 are supported.")
    endif()
endfunction()


# Function to download and extract piper-phonemize
function(download_piper_phonemize)
    detect_platform()
    set(PIPER_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/piper_phonemize")
    
    if(NOT EXISTS "${PIPER_DIR}/include")
        message(STATUS "Downloading piper-phonemize for ${PLATFORM_OS}-${PLATFORM_ARCH}")
        
        if(PLATFORM_OS STREQUAL "linux" AND PLATFORM_ARCH STREQUAL "x64")
            set(PIPER_URL "https://github.com/rhasspy/piper-phonemize/releases/download/2023.11.14-4/piper-phonemize_linux_x86_64.tar.gz")
            set(PIPER_ARCHIVE "${CMAKE_CURRENT_SOURCE_DIR}/deps/piper-phonemize_linux_x86_64.tar.gz")
        elseif(PLATFORM_OS STREQUAL "linux" AND PLATFORM_ARCH STREQUAL "arm64")
            set(PIPER_URL "https://github.com/rhasspy/piper-phonemize/releases/download/2023.11.14-4/piper-phonemize_linux_aarch64.tar.gz")
            set(PIPER_ARCHIVE "${CMAKE_CURRENT_SOURCE_DIR}/deps/piper-phonemize_linux_aarch64.tar.gz")
        else()
            message(FATAL_ERROR "Unsupported platform ${PLATFORM_OS}-${PLATFORM_ARCH} for piper-phonemize auto-download")
        endif()
        
        # Create deps directory if it doesn't exist
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/deps")
        
        # Download the archive using CMake's built-in download
        file(DOWNLOAD "${PIPER_URL}" "${PIPER_ARCHIVE}" 
             STATUS DOWNLOAD_STATUS
             SHOW_PROGRESS)
        list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
        
        if(DOWNLOAD_RESULT EQUAL 0)
            # Extract the archive (Linux uses .tar.gz files)
            execute_process(
                COMMAND tar -xzf "${PIPER_ARCHIVE}" -C "${CMAKE_CURRENT_SOURCE_DIR}/deps"
                RESULT_VARIABLE EXTRACT_RESULT
            )
            
            if(EXTRACT_RESULT EQUAL 0)
                # Clean up the archive
                file(REMOVE "${PIPER_ARCHIVE}")
                message(STATUS "Successfully downloaded and extracted piper-phonemize")
            else()
                message(FATAL_ERROR "Failed to extract piper-phonemize archive")
            endif()
        else()
            message(FATAL_ERROR "Failed to download piper-phonemize from ${PIPER_URL}")
        endif()
    else()
        message(STATUS "piper-phonemize already available at ${PIPER_DIR}")
    endif()

    include_directories(${PIPER_DIR}/include)
    link_directories(${PIPER_DIR}/lib)

    if(NOT DEFINED ESPEAK_NG_DATA_DIR)
        set(ESPEAK_NG_DATA_DIR "${PIPER_DIR}/share/espeak-ng-data" PARENT_SCOPE)
    else()
        message(STATUS "ESPEAK_NG_DATA_DIR already defined as ${ESPEAK_NG_DATA_DIR}")
    endif()
endfunction()

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
    message(FATAL_ERROR "nlohmann/json is required for configuration file loading.")
endif()

# Set default build type to Release for performance
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add optimization flags based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    # Release optimizations
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -DNDEBUG")
    
    # Smart CPU optimization detection
    if(CMAKE_CROSSCOMPILING)
        # Cross-compiling: use generic optimizations for target architecture
        if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
            if(AARCH64_MCPU)
                set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mcpu=${AARCH64_MCPU}")
                message(STATUS "Building in Release mode with ARM64 optimizations: -mcpu=${AARCH64_MCPU} (cross-compiling)")
            else()
                message(STATUS "Building in Release mode with generic ARM64 optimizations (cross-compiling)")
            endif()
        elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
            set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=x86-64")
            message(STATUS "Building in Release mode with x86_64 optimizations (cross-compiling)")
        else()
            message(STATUS "Building in Release mode with generic optimizations (cross-compiling)")
        endif()
    else()
        # Native compilation: use native CPU features
        set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native")
        message(STATUS "Building in Release mode with native CPU optimizations")
    endif()
    
elseif(CMAKE_BUILD_TYPE STREQUAL "Debug")
    # Debug flags
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0")
    message(STATUS "Building in Debug mode")
endif()

# stt, tts, and llm options
option(USE_WHISPER   "Enable whisper.cpp for STT"   OFF)
option(USE_SHERPA    "Enable sherpa-onnx for STT"   OFF)
option(USE_RKLLM     "Enable rkllm for LLM"         OFF)
option(USE_LLAMA     "Enable llama.cpp for LLM"     OFF)
option(USE_PAROLI    "Enable Paroli for TTS"        OFF)

# Statistics and monitoring options
option(ENABLE_STATS_LOGGING "Enable pipeline statistics logging and monitoring" OFF)

find_package(PkgConfig REQUIRED)
find_package(SDL2 REQUIRED)
find_package(ALSA REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SOURCES
    src/main.cpp
    src/common.cpp
    src/async_pipeline_factory.cpp
    src/async_processors.cpp
)

# Statistics logging compile definition
if(ENABLE_STATS_LOGGING)
    add_definitions(-DENABLE_STATS_LOGGING)
endif()

if(USE_WHISPER)
    add_compile_definitions(STT_BACKEND=WhisperSTT)
    add_definitions(-DUSE_WHISPER)
    list(APPEND SOURCES
        src/common-sdl.cpp
        src/common.cpp
        src/stt_whisper.cpp
    )
    add_subdirectory(third_party/whisper.cpp ${CMAKE_BINARY_DIR}/whisper-build EXCLUDE_FROM_ALL)
    list(APPEND EXTRA_LIBS whisper)
elseif(USE_SHERPA)
    add_compile_definitions(STT_BACKEND=SherpaSTT)
    add_definitions(-DUSE_SHERPA)

    add_definitions(-DSHERPA_ONNX_ENABLE_ALSA)
    list(APPEND SOURCES
        src/stt_sherpa.cpp
        third_party/sherpa-onnx/sherpa-onnx/csrc/alsa.cc
    )

    # Disable unused features
    set(SHERPA_ONNX_ENABLE_TTS OFF CACHE BOOL "Disable sherpa-onnx TTS (handled by Paroli)" FORCE)
    set(SHERPA_ONNX_ENABLE_SPEAKER_DIARIZATION OFF CACHE BOOL "Disable sherpa-onnx speaker diarization (unused)" FORCE)
    
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/sherpa-onnx)
    add_subdirectory(third_party/sherpa-onnx ${CMAKE_BINARY_DIR}/sherpa-onnx-build EXCLUDE_FROM_ALL)
    list(APPEND EXTRA_LIBS sherpa-onnx-cxx-api)
else()
    message(FATAL_ERROR "No STT backend selected")
endif()

if(USE_LLAMA)
    add_compile_definitions(LLM_BACKEND=LlamaLLM)
    add_definitions(-DUSE_LLAMA)
    list(APPEND SOURCES src/llm_llama.cpp)
    add_subdirectory(third_party/llama.cpp ${CMAKE_BINARY_DIR}/llama-build EXCLUDE_FROM_ALL)
    list(APPEND EXTRA_LIBS llama)
elseif(USE_RKLLM)
    # Link RKLLM from file system
    set(RKLLM_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/rknn-llm/rkllm-runtime/Linux")
    
    # Add include directories
    include_directories(${RKLLM_DIR}/librkllm_api/include)
    
    # Add library directory (architecture-specific)
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        link_directories(${RKLLM_DIR}/librkllm_api/aarch64)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "armhf")
        link_directories(${RKLLM_DIR}/librkllm_api/armhf)
    else()
        message(FATAL_ERROR "Unsupported architecture for RKLLM: ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
    
    add_compile_definitions(LLM_BACKEND=RknnLLM)
    add_definitions(-DUSE_RKLLM)
    list(APPEND SOURCES src/llm_rknn.cpp)
else()
    message(FATAL_ERROR "No LLM backend selected")
endif()

if(USE_PAROLI)
    # Automatically download piper-phonemize (includes ONNX Runtime 1.14.1)
    download_piper_phonemize()

    add_compile_definitions(ESPEAK_NG_DATA_DIR="${ESPEAK_NG_DATA_DIR}")
    
    # Ensure CMake can find custom Find*.cmake modules provided by paroli-daemon
    list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/paroli-daemon/cmake")

    add_subdirectory(third_party/paroli-daemon ${CMAKE_BINARY_DIR}/paroli-build EXCLUDE_FROM_ALL)

    link_directories(${CMAKE_BINARY_DIR}/_deps/espeak_ng-build)
    list(APPEND EXTRA_LIBS paroli-daemon-lib)

    # Add include directories for paroli-daemon
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/paroli-daemon)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/paroli-daemon/paroli-daemon)

    # Add compile definitions
    add_compile_definitions(TTS_BACKEND=TTSParoli)
    add_definitions(-DUSE_PAROLI)
    list(APPEND SOURCES src/tts_paroli.cpp)
else()
    message(FATAL_ERROR "No TTS backend selected")
endif()

# Link libraries and directories
add_executable(local-llm ${SOURCES})
set_property(TARGET local-llm PROPERTY CXX_STANDARD 20)

target_include_directories(local-llm PRIVATE ${SDL2_INCLUDE_DIRS} ${ALSA_INCLUDE_DIRS} ${nlohmann_json_INCLUDE_DIRS})
target_link_libraries(local-llm PRIVATE ${EXTRA_LIBS} ${SDL2_LIBRARIES} ${ALSA_LIBRARIES} pthread)
target_link_directories(local-llm PRIVATE
    ${CMAKE_BINARY_DIR}/lib
)
if(USE_LLAMA)
    target_link_libraries(local-llm PRIVATE llama)
elseif(USE_RKLLM)
    target_link_libraries(local-llm PRIVATE rkllmrt)
endif()
if(USE_WHISPER)
    target_link_libraries(local-llm PRIVATE whisper)
endif()
if(USE_SHERPA)
    target_link_libraries(local-llm PRIVATE sherpa-onnx-cxx-api)
endif()
if(USE_PAROLI)
    target_include_directories(local-llm PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/paroli-daemon
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/paroli-daemon/paroli-daemon)
    target_link_libraries(local-llm PRIVATE onnxruntime piper_phonemize)
endif()