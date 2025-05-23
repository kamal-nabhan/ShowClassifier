#include "openai_client.h"
#include <iostream>
#include <stdexcept> // For std::runtime_error

// CPR for HTTP requests - ensure you have this library available and linked
// #define CPR_USE_SYSTEM_CURL // Optional: if you want CPR to use system's libcurl
#include <cpr/cpr.h>

// nlohmann/json for JSON manipulation - ensure you have this library available
#include <nlohmann/json.hpp>
using json = nlohmann::json;


OpenAIClient::OpenAIClient(TranscriptContextBuilder& context_builder, std::string api_key, const std::string& model_name)
    : context_builder_(context_builder), 
      api_key_(std::move(api_key)),
      model_name_(model_name) {
    if (api_key_.empty()) {
        std::cerr << "Warning: OpenAI API key is empty. Classification will fail." << std::endl;
    }
}

OpenAIClient::~OpenAIClient() {
    stop_classification_loop();
}

void OpenAIClient::start_periodic_classification(std::chrono::seconds interval) {
    if (running_) {
        std::cout << "OpenAIClient classification loop is already running." << std::endl;
        return;
    }
    if (api_key_.empty()) {
        std::cerr << "Cannot start OpenAIClient: API key is empty." << std::endl;
        return;
    }
    classification_interval_ = interval;
    running_ = true;
    classification_thread_ = std::thread(&OpenAIClient::periodic_classification_loop, this);
    std::cout << "OpenAIClient periodic classification started with interval: " << interval.count() << "s." << std::endl;
}

void OpenAIClient::stop_classification_loop() {
    if (!running_) {
        return;
    }
    running_ = false;
    if (classification_thread_.joinable()) {
        classification_thread_.join();
    }
    std::cout << "OpenAIClient classification loop stopped." << std::endl;
}

std::string OpenAIClient::get_last_classification_result() const {
    std::lock_guard<std::mutex> lock(result_mutex_);
    return last_classification_result_;
}

void OpenAIClient::periodic_classification_loop() {
    while (running_) {
        // Wait for the specified interval
        // This can be done more accurately with a condition variable if precise timing and immediate exit are needed
        std::this_thread::sleep_for(classification_interval_);
        if (!running_) break; // Check again after sleep

        std::string current_transcript = context_builder_.get_full_transcript();
        if (current_transcript.empty() || current_transcript.find_first_not_of(" \n\r\t") == std::string::npos) {
            // std::cout << "[OpenAIClient] Transcript is empty, skipping classification." << std::endl;
            continue;
        }

        // std::cout << "[OpenAIClient] Processing transcript of length: " << current_transcript.length() << std::endl;
        std::string classification_result = classify_text_with_openai(current_transcript);
        
        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            last_classification_result_ = classification_result;
        }
        std::cout << "==> OpenAI Classification: " << classification_result << std::endl;
    }
}

std::string OpenAIClient::classify_text_with_openai(const std::string& transcript_text) {
    std::cout << transcript_text << std::endl;
    // Prepare the JSON payload for the OpenAI API request
    json payload = {
        {"model", model_name_},
        {"messages", json::array({
            {{"role", "system"}, {"content", system_prompt_}},
            {{"role", "user"}, {"content", transcript_text}}
        })},
        {"temperature", 0.2}, // Lower temperature for more deterministic classification
        {"max_tokens", 60}    // Max tokens for the movie/show title, adjust as needed
    };

    try {
        // Set a timeout for the request (e.g., 15 seconds)
        cpr::Response r = cpr::Post(cpr::Url{"https://api.openai.com/v1/chat/completions"},
                                    cpr::Bearer{api_key_},
                                    cpr::Header{{"Content-Type", "application/json"}},
                                    cpr::Body{payload.dump()},
                                    cpr::Timeout{15000}); // 15 seconds timeout in milliseconds

        if (r.status_code == 200) {
            return parse_openai_json_response(r.text);
        } else {
            std::cerr << "[OpenAIClient] API Error " << r.status_code << ": " << r.text << std::endl;
            if (r.status_code == 401) return "Unknown (Invalid API Key or Auth Error)";
            if (r.status_code == 429) return "Unknown (Rate Limit Exceeded)";
            return std::string("Unknown (API Error ") + std::to_string(r.status_code) + ")";
        }
    } catch (const std::exception& e) {
        std::cerr << "[OpenAIClient] HTTP Request Exception: " << e.what() << std::endl;
        return "Unknown (Request Exception)";
    }
}

std::string OpenAIClient::parse_openai_json_response(const std::string& json_response_str) {
    try {
        json response_json = json::parse(json_response_str);
        if (response_json.contains("choices") && 
            response_json["choices"].is_array() && 
            !response_json["choices"].empty()) {
            
            const auto& first_choice = response_json["choices"][0];
            if (first_choice.contains("message") && 
                first_choice["message"].contains("content")) {
                return first_choice["message"]["content"].get<std::string>();
            }
        }
        std::cerr << "[OpenAIClient] Could not find 'content' in choices[0].message. Response: " << json_response_str << std::endl;
        return "Unknown (Invalid Response Format)";
    } catch (const json::parse_error& e) {
        std::cerr << "[OpenAIClient] JSON parse error: " << e.what() << ". Response was: " << json_response_str << std::endl;
        return "Unknown (Response Parse Error)";
    } catch (const std::exception& e) {
        std::cerr << "[OpenAIClient] Error parsing JSON: " << e.what() << std::endl;
        return "Unknown (Generic Parse Error)";
    }
}