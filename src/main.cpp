#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <thread>
#include <csignal> 
#include <atomic>

#include "concurrent_queue.h"
#include "audio_capturer.h"
#include "transcript_context_builder.h"
#include "whisper_transcriber.h"
#include "openai_client.h"

std::atomic<bool> g_application_running(true);

void signal_handler(int signum) {
    std::cout << std::endl << "[Main] Interrupt signal (" << signum << ") received. Shutting down..." << std::endl;
    g_application_running = false;
}

int main(int argc, char* argv[]) {
    std::cout << "[Main] Application starting..." << std::endl;
    if (argc < 2) { 
        std::cerr << "[Main] Usage: " << argv[0] << " <path_to_whisper_ggml_model.bin> [language_code (e.g., en, auto)]" << std::endl;
        std::cerr << "[Main] Example: " << argv[0] << " ./models/ggml-base.en.bin en" << std::endl;
        std::cerr << "[Main] OpenAI API Key and endpoint are hardcoded." << std::endl;
        return 1;
    }

    std::string model_path = argv[1];
    std::cout << "[Main] Whisper model path: " << model_path << std::endl;

    std::string openai_api_key = "2z2lTmn0nUX7BiJVFfOqt8E8Nwns9vj9sonjkNKmknhKXaXOk1h2JQQJ99BEACHYHv6XJ3w3AAABACOG4FAN";
    std::string openai_endpoint = "https://hackfest25-openai-23.openai.azure.com/";
    std::string deployment_name = "gpt-4o-mini";
    std::string api_version = "2024-12-01-preview";

    std::string language = "auto"; 
    if (argc >= 3) { 
        language = argv[2];
    }
    std::cout << "[Main] Transcription language: " << language << std::endl;
    std::chrono::seconds openai_call_interval(10); 

    signal(SIGINT, signal_handler);  
    signal(SIGTERM, signal_handler); 

    try {
        std::cout << "[Main] Initializing components..." << std::endl;
        ConcurrentQueue<std::vector<float>> audio_data_queue;
        TranscriptContextBuilder context_builder;

        AudioCapturer audio_capturer(audio_data_queue);
        WhisperTranscriber whisper_transcriber(model_path, language, audio_data_queue, context_builder);
        OpenAIClient openai_client(context_builder, openai_api_key, openai_endpoint, deployment_name, api_version);

        std::cout << "[Main] Starting components..." << std::endl;
        if (!audio_capturer.start_stream()) {
            std::cerr << "[Main] ERROR: Failed to start audio capturer. Exiting." << std::endl;
            return 1;
        }

        if (!whisper_transcriber.start()) {
            std::cerr << "[Main] ERROR: Failed to start whisper transcriber. Exiting." << std::endl;
            audio_capturer.stop_stream(); 
            return 1;
        }
        
        openai_client.start_periodic_classification(openai_call_interval);

        std::cout << "[Main] System initialized. Capturing audio and transcribing..." << std::endl;
        std::cout << "[Main] Using OpenAI endpoint: " << openai_endpoint << " with deployment: " << deployment_name << std::endl;
        std::cout << "[Main] Press Ctrl+C to exit." << std::endl;

        std::string last_printed_classification = "Unknown"; // To avoid spamming console
        while (g_application_running) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            // Optionally, print classification if it changes
            std::string current_classification = openai_client.get_last_classification_result();
            if (current_classification != last_printed_classification && current_classification != "Unknown (API Error 429)" && !current_classification.empty() && current_classification != "Unknown (Request Exception)" && current_classification.rfind("Unknown (API Error", 0) != 0) {
               if (last_printed_classification.rfind("Unknown (API Error", 0) == 0 && current_classification == "Unknown (Rate Limit Exceeded)") {
                    // Avoid printing rate limit if already showing an API error
               } else {
                    std::cout << "[Main] Current Classification: " << current_classification << std::endl;
                    last_printed_classification = current_classification;
               }
            } else if (current_classification.rfind("Unknown (API Error", 0) == 0 && current_classification != last_printed_classification) {
                 std::cout << "[Main] Current API Status: " << current_classification << std::endl;
                 last_printed_classification = current_classification;
            }

        }

        std::cout << "[Main] Main loop terminated. Stopping components..." << std::endl;
        
        openai_client.stop_classification_loop(); 
        whisper_transcriber.stop();        
        audio_capturer.stop_stream();      

        std::cout << "[Main] All components stopped. Exiting." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[Main] FATAL UNHANDLED EXCEPTION: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "[Main] FATAL UNKNOWN UNHANDLED EXCEPTION." << std::endl;
        return 1;
    }

    std::cout << "[Main] Application finished successfully." << std::endl;
    return 0;
}