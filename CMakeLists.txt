cmake_minimum_required(VERSION 3.16)
project(local-llm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Function to detect OS and architecture for dependency downloads
function(detect_platform)
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
        set(PLATFORM_OS "linux" PARENT_SCOPE)
    elseif(CMAKE_SYSTEM_NAME STREQUAL "Windows")
        set(PLATFORM_OS "win" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Unsupported OS: ${CMAKE_SYSTEM_NAME}. Only Linux and Windows are supported.")
    endif()
    
    if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
        set(PLATFORM_ARCH "x64" PARENT_SCOPE)
    elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
        set(PLATFORM_ARCH "arm64" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}. Only x86_64 and arm64 are supported.")
    endif()
endfunction()

# Function to download and extract ONNX Runtime
function(download_onnxruntime)
    detect_platform()
    set(ORT_VERSION "1.15.0")
    set(ORT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime-${PLATFORM_OS}-${PLATFORM_ARCH}-${ORT_VERSION}")
    
    if(NOT EXISTS "${ORT_DIR}/include")
        message(STATUS "Downloading ONNX Runtime ${ORT_VERSION} for ${PLATFORM_OS}-${PLATFORM_ARCH}")
        
        if(PLATFORM_OS STREQUAL "linux" AND PLATFORM_ARCH STREQUAL "x64")
            set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-${ORT_VERSION}.tgz")
            set(ORT_ARCHIVE "${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime-linux-x64-${ORT_VERSION}.tgz")
        elseif(PLATFORM_OS STREQUAL "linux" AND PLATFORM_ARCH STREQUAL "arm64")
            set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-aarch64-${ORT_VERSION}.tgz")
            set(ORT_ARCHIVE "${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime-linux-aarch64-${ORT_VERSION}.tgz")
        elseif(PLATFORM_OS STREQUAL "win" AND PLATFORM_ARCH STREQUAL "x64")
            set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-win-x64-${ORT_VERSION}.zip")
            set(ORT_ARCHIVE "${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime-win-x64-${ORT_VERSION}.zip")
        elseif(PLATFORM_OS STREQUAL "win" AND PLATFORM_ARCH STREQUAL "arm64")
            set(ORT_URL "https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-win-arm64-${ORT_VERSION}.zip")
            set(ORT_ARCHIVE "${CMAKE_CURRENT_SOURCE_DIR}/deps/onnxruntime-win-arm64-${ORT_VERSION}.zip")
        else()
            message(FATAL_ERROR "Unsupported platform ${PLATFORM_OS}-${PLATFORM_ARCH} for ONNX Runtime auto-download")
        endif()
        
        # Create deps directory if it doesn't exist
        file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/deps")
        
        # Download the archive using CMake's built-in download
        file(DOWNLOAD "${ORT_URL}" "${ORT_ARCHIVE}" 
             STATUS DOWNLOAD_STATUS
             SHOW_PROGRESS)
        list(GET DOWNLOAD_STATUS 0 DOWNLOAD_RESULT)
        
        if(DOWNLOAD_RESULT EQUAL 0)
            # Extract the archive based on platform
            if(PLATFORM_OS STREQUAL "linux")
                execute_process(
                    COMMAND tar -xzf "${ORT_ARCHIVE}" -C "${CMAKE_CURRENT_SOURCE_DIR}/deps"
                    RESULT_VARIABLE EXTRACT_RESULT
                )
            elseif(PLATFORM_OS STREQUAL "win")
                execute_process(
                    COMMAND ${CMAKE_COMMAND} -E tar xzf "${ORT_ARCHIVE}" -C "${CMAKE_CURRENT_SOURCE_DIR}/deps"
                    RESULT_VARIABLE EXTRACT_RESULT
                )
            endif()
            
            if(EXTRACT_RESULT EQUAL 0)
                # Clean up the archive
                file(REMOVE "${ORT_ARCHIVE}")
                message(STATUS "Successfully downloaded and extracted ONNX Runtime")
            else()
                message(FATAL_ERROR "Failed to extract ONNX Runtime archive")
            endif()
        else()
            message(FATAL_ERROR "Failed to download ONNX Runtime from ${ORT_URL}")
        endif()
    else()
        message(STATUS "ONNX Runtime already available at ${ORT_DIR}")
    endif()
    
    set(ORT_ROOT "${ORT_DIR}" PARENT_SCOPE)
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
        elseif(PLATFORM_OS STREQUAL "win" AND PLATFORM_ARCH STREQUAL "x64")
            set(PIPER_URL "https://github.com/rhasspy/piper-phonemize/releases/download/2023.11.14-4/piper-phonemize_windows_amd64.zip")
            set(PIPER_ARCHIVE "${CMAKE_CURRENT_SOURCE_DIR}/deps/piper-phonemize_windows_amd64.zip")
        elseif(PLATFORM_OS STREQUAL "win" AND PLATFORM_ARCH STREQUAL "arm64")
            set(PIPER_URL "https://github.com/rhasspy/piper-phonemize/releases/download/2023.11.14-4/piper-phonemize_windows_arm64.zip")
            set(PIPER_ARCHIVE "${CMAKE_CURRENT_SOURCE_DIR}/deps/piper-phonemize_windows_arm64.zip")
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
            # Extract the archive based on platform
            if(PLATFORM_OS STREQUAL "linux")
                execute_process(
                    COMMAND tar -xzf "${PIPER_ARCHIVE}" -C "${CMAKE_CURRENT_SOURCE_DIR}/deps"
                    RESULT_VARIABLE EXTRACT_RESULT
                )
            elseif(PLATFORM_OS STREQUAL "win")
                execute_process(
                    COMMAND ${CMAKE_COMMAND} -E tar xzf "${PIPER_ARCHIVE}" -C "${CMAKE_CURRENT_SOURCE_DIR}/deps"
                    RESULT_VARIABLE EXTRACT_RESULT
                )
            endif()
            
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
    
    set(PIPER_PHONEMIZE_ROOT "${PIPER_DIR}" PARENT_SCOPE)
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

# set(CMAKE_BUILD_TYPE Debug)

# stt, tts, and llm options
option(USE_WHISPER   "Enable whisper.cpp for STT"   ON)
option(USE_RKLLM     "Enable rkllm for LLM"         OFF)
option(USE_LLAMA     "Enable llama.cpp for LLM"     ON)
option(USE_Paroli    "Enable Paroli for TTS"        ON)

find_package(SDL2 REQUIRED)
find_package(ALSA REQUIRED)

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

set(SOURCES
    src/main.cpp
    src/common-sdl.cpp
    src/common.cpp
    src/async_pipeline_factory.cpp
    src/async_processors.cpp
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

# Add paroli-daemon subdirectory if using Paroli TTS
if(USE_Paroli)
    # Automatically download ONNX Runtime and piper-phonemize if not available
    download_onnxruntime()
    download_piper_phonemize()
    
    # Set the paths (these are set by the download functions)
    set(ORT_ROOT "${ORT_ROOT}" CACHE STRING "Path to ONNX Runtime root directory")
    set(PIPER_PHONEMIZE_ROOT "${PIPER_PHONEMIZE_ROOT}" CACHE STRING "Path to piper-phonemize root directory (used by Paroli)")

    # Ensure CMake can find custom Find*.cmake modules provided by paroli-daemon
    list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third_party/paroli-daemon/cmake")

    add_subdirectory(third_party/paroli-daemon ${CMAKE_BINARY_DIR}/paroli-build EXCLUDE_FROM_ALL)
    list(APPEND EXTRA_LIBS paroli-daemon-lib)

    # Add include directories for paroli-daemon
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/paroli-daemon)
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party/paroli-daemon/paroli-daemon)
endif()

if(USE_Paroli)
    add_compile_definitions(TTS_BACKEND=TTSParoli)
    add_definitions(-DUSE_Paroli)
    list(APPEND SOURCES src/tts_paroli.cpp)
else()
    message(FATAL_ERROR "No TTS backend selected")
endif()

add_executable(local-llm ${SOURCES})
target_include_directories(local-llm PRIVATE ${SDL2_INCLUDE_DIRS} ${ALSA_INCLUDE_DIRS} ${nlohmann_json_INCLUDE_DIRS})
# Ensure Paroli headers (and ORT/piper-phonemize headers) are visible when compiling local sources
if(USE_Paroli)
    target_include_directories(local-llm PRIVATE
        ${ORT_ROOT}/include
        ${PIPER_PHONEMIZE_ROOT}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/paroli-daemon
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/paroli-daemon/paroli-daemon)
endif()
target_link_libraries(local-llm PRIVATE whisper llama ${EXTRA_LIBS} ${SDL2_LIBRARIES} ${ALSA_LIBRARIES} pthread)
if(USE_Paroli)
    # Ensure ONNX Runtime is linked and library paths are available to the final executable
    target_link_directories(local-llm PRIVATE ${ORT_ROOT}/lib ${PIPER_PHONEMIZE_ROOT}/lib)
    target_link_libraries(local-llm PRIVATE onnxruntime)
    # Optionally embed rpath so running without env vars works
    set_target_properties(local-llm PROPERTIES
        BUILD_RPATH "${ORT_ROOT}/lib;${PIPER_PHONEMIZE_ROOT}/lib"
        INSTALL_RPATH "${ORT_ROOT}/lib;${PIPER_PHONEMIZE_ROOT}/lib")
endif()
