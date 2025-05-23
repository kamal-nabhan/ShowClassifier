cmake_minimum_required(VERSION 3.15)
project(RealtimeShowClassifier LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- Whisper.cpp ---
add_subdirectory(whisper.cpp)

# --- SDL2 (for Audio Capture) ---
# This will find SDL2. If it's not found, CMake will error out.
# SDL2::SDL2 is the imported target for the main SDL2 library.
# SDL2::main is often needed for cross-platform main function handling with SDL.
find_package(SDL2 REQUIRED)

# --- CPR & nlohmann/json (via FetchContent) ---
include(FetchContent)

FetchContent_Declare(
  nlohmann_json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.3
)

FetchContent_MakeAvailable(nlohmann_json)

FetchContent_Declare(
  cpr_source
  GIT_REPOSITORY https://github.com/libcpr/cpr.git
  GIT_TAG 1.10.5
)
FetchContent_MakeAvailable(cpr_source)
# CPR might also need libcurl, OpenSSL depending on your system and CPR version.
# find_package(CURL REQUIRED)
# find_package(OpenSSL REQUIRED)

# --- Threads ---
find_package(Threads REQUIRED)

# --- Your Executable ---
add_executable(ShowClassifier
    src/main.cpp
    src/audio_capturer.cpp src/audio_capturer.h
    src/whisper_transcriber.cpp src/whisper_transcriber.h
    src/transcript_context_builder.cpp src/transcript_context_builder.h
    src/openai_client.cpp src/openai_client.h
    # concurrent_queue.h is header-only, included directly
)

target_include_directories(ShowClassifier PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${CMAKE_CURRENT_SOURCE_DIR}/whisper.cpp # For whisper.h
    ${SDL2_INCLUDE_DIRS}                    # Include directory for SDL2
    ${cpr_SOURCE_DIR}/include              # Include directory for CPR
    # nlohmann/json is often header-only and its include path is handled by its CMake target
)

target_link_libraries(ShowClassifier PRIVATE
    whisper                        # from whisper.cpp
    # common
    # common-sdl
    SDL2::SDL2                     # Main SDL2 library target
    # SDL2::Main                     # SDL2 main handling library (important for cross-platform)
    cpr::cpr                       # from CPR
    nlohmann_json::nlohmann_json   # from nlohmann/json
    Threads::Threads               # For std::thread
    # Link CURL and OpenSSL if CPR doesn't do it automatically on your system
    # You might need to add find_package(CURL REQUIRED) and find_package(OpenSSL REQUIRED)
    # if they are not found as dependencies of CPR.
    # For example:
    # CURL::libcurl
    # OpenSSL::SSL
    # OpenSSL::Crypto
)

# Ensure C++17 filesystem is available if needed by any part of the code.
# This is usually handled by modern compilers with CMAKE_CXX_STANDARD 17.
# If you encounter issues with std::filesystem, uncomment and adapt:
# if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8)
#    target_link_libraries(ShowClassifier PRIVATE stdc++fs)
# elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 7)
#    target_link_libraries(ShowClassifier PRIVATE stdc++fs)
# endif()
