#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <csignal> // For signal handling (Ctrl+C)
#include <atomic>

#include "concurrent_queue.h"
#include "audio_capturer.h"
#include "transcript_context_builder.h"
#include "whisper_transcriber.h"
#include "openai_client.h"

// Global flag to signal termination for all threads
std::atomic<bool> g_application_running(true);

void signal_handler(int signum) {
    std::cout << std::endl << "Interrupt signal (" << signum << ") received. Shutting down..." << std::endl;
    g_application_running = false;
}

int main(int argc, char* argv[]) {
    // --- Configuration ---
    if (argc < 2) { // MODIFICATION 2: API key is now hardcoded, so only model path is essential from args
        std::cerr << "Usage: " << argv[0] << " <path_to_whisper_ggml_model.bin> [language_code (e.g., en, auto)]" << std::endl;
        std::cerr << "Example: " << argv[0] << " ./models/ggml-base.en.bin en" << std::endl;
        std::cerr << "OpenAI API Key and endpoint are now hardcoded in main.cpp." << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    // MODIFICATION 2: Hardcode Azure OpenAI credentials
    std::string openai_api_key = "2z2lTmn0nUX7BiJVFfOqt8E8Nwns9vj9sonjkNKmknhKXaXOk1h2JQQJ99BEACHYHv6XJ3w3AAABACOG4FAN";
    std::string openai_endpoint = "https://hackfest25-openai-23.openai.azure.com/";
    std::string deployment_name = "gpt-4o-mini";
    std::string api_version = "2024-12-01-preview";

    std::string language = "auto"; // Default to auto-detect
    if (argc >= 3) { // Adjusted arg index since API key is no longer argv[2]
        language = argv[2];
    }
    std::chrono::seconds openai_call_interval(10); // How often to call OpenAI

    // --- Setup signal handler for graceful shutdown ---
    signal(SIGINT, signal_handler);  // Handle Ctrl+C
    signal(SIGTERM, signal_handler); // Handle termination signal

    // --- Initialize Components ---
    try {
        ConcurrentQueue<std::vector<float>> audio_data_queue;
        TranscriptContextBuilder context_builder;

        AudioCapturer audio_capturer(audio_data_queue);
        WhisperTranscriber whisper_transcriber(model_path, language, audio_data_queue, context_builder);
        
        // MODIFICATION 2: Update OpenAIClient instantiation with new parameters
        OpenAIClient openai_client(context_builder, openai_api_key, openai_endpoint, deployment_name, api_version);

        // --- Start Components ---
        if (!audio_capturer.start_stream()) {
            std::cerr << "Failed to start audio capturer. Exiting." << std::endl;
            return 1;
        }

        if (!whisper_transcriber.start()) {
            std::cerr << "Failed to start whisper transcriber. Exiting." << std::endl;
            audio_capturer.stop_stream(); // Clean up audio capturer
            return 1;
        }
        
        openai_client.start_periodic_classification(openai_call_interval);

        std::cout << "System initialized. Capturing audio and transcribing..." << std::endl;
        std::cout << "Using OpenAI endpoint: " << openai_endpoint << " with deployment: " << deployment_name << std::endl;
        std::cout << "Press Ctrl+C to exit." << std::endl;

        // --- Main Loop (Keep application alive) ---
        while (g_application_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }

        // --- Shutdown Components (in reverse order of starting if dependencies exist) ---
        std::cout << "Main loop terminated. Stopping components..." << std::endl;
        
        openai_client.stop_classification_loop(); 
        whisper_transcriber.stop();        
        audio_capturer.stop_stream();      

        std::cout << "All components stopped. Exiting." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "Unhandled exception in main: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown unhandled exception in main." << std::endl;
        return 1;
    }

    return 0;
}