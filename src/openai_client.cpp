#include "openai_client.h"
#include <iostream>
#include <stdexcept> 
#include <string_view> 

#include <cpr/cpr.h>
#include <nlohmann/json.hpp>
using json = nlohmann::json;

OpenAIClient::OpenAIClient(TranscriptContextBuilder& context_builder, 
                           std::string api_key, 
                           const std::string& openai_endpoint,
                           const std::string& deployment_name,
                           const std::string& api_version)
    : context_builder_(context_builder), 
      api_key_(std::move(api_key)),
      openai_endpoint_(openai_endpoint),
      deployment_name_(deployment_name),
      api_version_(api_version) {
    std::cout << "[OpenAIClient] Initializing..." << std::endl;
    if (api_key_.empty()) {
        std::cerr << "[OpenAIClient] Warning: OpenAI API key is empty." << std::endl;
    }
    if (openai_endpoint_.empty()) {
        std::cerr << "[OpenAIClient] Warning: OpenAI endpoint is empty." << std::endl;
    } else {
        std::cout << "[OpenAIClient] Endpoint: " << openai_endpoint_ << std::endl;
    }
    if (deployment_name_.empty()) {
        std::cerr << "[OpenAIClient] Warning: OpenAI deployment name is empty." << std::endl;
    } else {
        std::cout << "[OpenAIClient] Deployment: " << deployment_name_ << std::endl;
    }
    if (api_version_.empty()) {
        std::cerr << "[OpenAIClient] Warning: OpenAI API version is empty." << std::endl;
    } else {
         std::cout << "[OpenAIClient] API Version: " << api_version_ << std::endl;
    }
     // The system_prompt_ is already defined in the header
     // std::cout << "[OpenAIClient DEBUG] System Prompt: " << system_prompt_ << std::endl; // Can be very verbose
    std::cout << "[OpenAIClient] Initialization complete." << std::endl;
}

OpenAIClient::~OpenAIClient() {
    std::cout << "[OpenAIClient] Destructing..." << std::endl;
    stop_classification_loop();
}

void OpenAIClient::start_periodic_classification(std::chrono::seconds interval) {
    if (running_) {
        std::cout << "[OpenAIClient] Classification loop is already running." << std::endl;
        return;
    }
    if (api_key_.empty() || openai_endpoint_.empty() || deployment_name_.empty() || api_version_.empty()) {
        std::cerr << "[OpenAIClient] ERROR: Cannot start. API key, endpoint, deployment name, or API version is empty." << std::endl;
        return;
    }
    classification_interval_ = interval;
    running_ = true;
    classification_thread_ = std::thread(&OpenAIClient::periodic_classification_loop, this);
    std::cout << "[OpenAIClient] Periodic classification started with interval: " << interval.count() << "s." << std::endl;
}

void OpenAIClient::stop_classification_loop() {
    if (!running_) {
        return;
    }
    std::cout << "[OpenAIClient] Stopping classification loop..." << std::endl;
    running_ = false;
    if (classification_thread_.joinable()) {
        classification_thread_.join();
    }
    std::cout << "[OpenAIClient] Classification loop stopped." << std::endl;
}

std::string OpenAIClient::get_last_classification_result() const {
    std::lock_guard<std::mutex> lock(result_mutex_);
    return last_classification_result_;
}

void OpenAIClient::periodic_classification_loop() {
    std::cout << "[OpenAIClient] Periodic classification loop started." << std::endl;
    while (running_) {
        std::this_thread::sleep_for(classification_interval_);
        if (!running_) {
            //std::cout << "[OpenAIClient DEBUG] Loop terminating due to running_ flag." << std::endl;
            break;
        }

        std::cout << "[OpenAIClient] Tick for classification." << std::endl;
        std::string current_transcript = context_builder_.get_full_transcript();
        
        if (current_transcript.empty() || current_transcript.find_first_not_of(" \n\r\t") == std::string::npos) {
            std::cout << "[OpenAIClient] Transcript is empty or whitespace, skipping classification." << std::endl;
            continue;
        }

        std::cout << "[OpenAIClient] Current transcript for classification (length " << current_transcript.length() << "): \"" << current_transcript << "\"" << std::endl;
        
        std::string classification_result = classify_text_with_openai(current_transcript);
        
        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            last_classification_result_ = classification_result;
        }
        // This is already printed in main.cpp or could be more structured if needed
        // std::cout << "==> OpenAI Classification: " << classification_result << std::endl; 
    }
    std::cout << "[OpenAIClient] Periodic classification loop finished." << std::endl;
}

std::string OpenAIClient::classify_text_with_openai(const std::string& transcript_text) {
    std::cout << "[OpenAIClient] Attempting to classify transcript..." << std::endl;
    //std::cout << "[OpenAIClient DEBUG] System prompt for API call: " << system_prompt_ << std::endl; // Can be verbose
    //std::cout << "[OpenAIClient DEBUG] User transcript for API call: " << transcript_text << std::endl;

    json payload = {
        {"messages", json::array({
            {{"role", "system"}, {"content", system_prompt_}},
            {{"role", "user"}, {"content", transcript_text}} 
        })},
        {"temperature", 0.2},
        {"max_tokens", 150} 
    };

    std::string request_url = openai_endpoint_;
    if (!request_url.empty() && request_url.back() != '/') { // Add trailing slash if missing and not empty
        request_url += '/';
    }
    request_url += "openai/deployments/" + deployment_name_ + "/chat/completions?api-version=" + api_version_;
    
    std::cout << "[OpenAIClient] Request URL: " << request_url << std::endl;
    //std::cout << "[OpenAIClient DEBUG] Payload: " << payload.dump(2) << std::endl;


    try {
        std::cout << "[OpenAIClient] Sending POST request to OpenAI..." << std::endl;
        cpr::Response r = cpr::Post(cpr::Url{request_url},
                                    cpr::Header{{"Content-Type", "application/json"},
                                                {"api-key", api_key_}},
                                    cpr::Body{payload.dump()},
                                    cpr::Timeout{15000}); 

        std::cout << "[OpenAIClient] Received response. Status code: " << r.status_code << std::endl;
        //std::cout << "[OpenAIClient DEBUG] Response body: " << r.text << std::endl;


        if (r.status_code == 200) {
            std::string parsed_response = parse_openai_json_response(r.text);
            std::cout << "[OpenAIClient] API call successful. Parsed response: \"" << parsed_response << "\"" << std::endl;
            return parsed_response;
        } else {
            std::cerr << "[OpenAIClient] API Error " << r.status_code << ": " << r.text << std::endl;
            // The following lines are already in your original code, good for diagnostics
            if (r.status_code == 401) return "Unknown (Invalid API Key or Auth Error)";
            if (r.status_code == 404) return "Unknown (Endpoint or Deployment Not Found)";
            if (r.status_code == 429) return "Unknown (Rate Limit Exceeded)";
            return std::string("Unknown (API Error ") + std::to_string(r.status_code) + ")";
        }
    } catch (const std::exception& e) {
        std::cerr << "[OpenAIClient] HTTP Request Exception: " << e.what() << std::endl;
        return "Unknown (Request Exception)";
    }
}

std::string OpenAIClient::parse_openai_json_response(const std::string& json_response_str) {
    //std::cout << "[OpenAIClient DEBUG] Parsing JSON response: " << json_response_str << std::endl;
    try {
        json response_json = json::parse(json_response_str);
        if (response_json.contains("choices") && 
            response_json["choices"].is_array() && 
            !response_json["choices"].empty()) {
            
            const auto& first_choice = response_json["choices"][0];
            if (first_choice.contains("message") && 
                first_choice["message"].contains("content")) {
                std::string content = first_choice["message"]["content"].get<std::string>();
                //std::cout << "[OpenAIClient DEBUG] Successfully parsed content: " << content << std::endl;
                return content;
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