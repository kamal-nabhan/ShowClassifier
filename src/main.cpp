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
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <path_to_whisper_ggml_model.bin> <your_openai_api_key> [language_code (e.g., en, auto)]" << std::endl;
        std::cerr << "Example: " << argv[0] << " ./models/ggml-base.en.bin YOUR_API_KEY_HERE en" << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::string openai_api_key = argv[2];
    std::string language = "auto"; // Default to auto-detect
    if (argc >= 4) {
        language = argv[3];
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
        OpenAIClient openai_client(context_builder, openai_api_key); // Default model "gpt-3.5-turbo"

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
        std::cout << "Press Ctrl+C to exit." << std::endl;

        // --- Main Loop (Keep application alive) ---
        while (g_application_running) {
            // Can print status, or just sleep.
            // Example: print last classification every few seconds if it changes
            // static std::string last_printed_classification = "";
            // std::string current_classification = openai_client.get_last_classification_result();
            // if(current_classification != last_printed_classification) {
            //    std::cout << "Current Classification: " << current_classification << std::endl;
            //    last_printed_classification = current_classification;
            // }
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }

        // --- Shutdown Components (in reverse order of starting if dependencies exist) ---
        std::cout << "Main loop terminated. Stopping components..." << std::endl;
        
        openai_client.stop_classification_loop(); // Stop before transcriber if it relies on context
        whisper_transcriber.stop();        // Stop before audio capturer if it relies on queue
        audio_capturer.stop_stream();      // Stop audio capture last for its thread

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