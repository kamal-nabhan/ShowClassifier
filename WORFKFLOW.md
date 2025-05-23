# C++ Project: TV Show/Movie Identification from Audio (CPU-based)

This document outlines a plan to create a C++ application for identifying TV shows or movies from audio, focusing on CPU execution and leveraging `whisper.cpp` and a Large Language Model (LLM). It also covers how to approach fine-tuning with a specific dataset.

It's an exciting project! Let's break down how you can approach this.

---
## Project Overview 🚀

The core idea is to build a C++ application that:

* Listens to an audio stream.
* Transcribes the speech in the audio to text using `whisper.cpp`, supporting 6 languages.
* Classifies the transcribed text to identify the specific movie or TV show (and potentially episode) using an LLM running on the CPU.

You also want to know how to use your dataset (20 samples of 5-minute audio with transcripts for 3 movies and 6 TV shows in 6 different languages) for fine-tuning. This dataset totals 100 minutes of audio.

---
## Part 1: Speech-to-Text (STT) with `whisper.cpp` 🎤

`whisper.cpp` is an excellent choice because it's a C/C++ port of OpenAI's Whisper model, designed for efficient CPU inference.

### Active Audio Listening in C++
You'll need a C++ library to capture audio from the microphone. Popular choices include:
* **PortAudio**: Cross-platform.
* **RtAudio**: Another cross-platform option.
* **Platform-specific APIs**: WASAPI (Windows), CoreAudio (macOS), ALSA (Linux).

The captured audio data (typically PCM, e.g., 16-bit, 16kHz, mono) will be continuously fed into `whisper.cpp`. The `whisper.cpp` examples (like `stream`) provide a good starting point for how to process audio in chunks.

### Multilingual Transcription
Whisper models are inherently multilingual. You'll select a `whisper.cpp` compatible model (e.g., `ggml-base.bin`, `ggml-small.bin`). Larger models are more accurate but slower.
`whisper.cpp` allows you to specify the language or use auto-detection. For your 6 languages, ensure the chosen model supports them well.

---
## Part 2: Movie/TV Show Classification with an LLM 🧠

For running LLMs on the CPU in C++, `llama.cpp` is the leading solution. It supports various quantized model formats (GGUF) that are optimized for CPU execution.

* **Input**: The text transcribed by `whisper.cpp`.
* **Output**: The name of the movie or TV show.

### Classification Strategy

#### Few-Shot Prompting
You can provide the LLM with a prompt that includes a few examples of dialogue snippets and their corresponding show/movie titles. The LLM then tries to classify the new transcript based on these examples. This is good for initial testing.

**Prompt Example:**

The following are dialogues from TV shows. Identify the show for the new dialogue.

Dialogue: "Winter is coming."
Show: Game of Thrones

Dialogue: "How you doin'?"
Show: Friends

Dialogue: "{new_transcribed_dialogue_from_whisper}"
Show: ?


#### Fine-tuning an LLM (discussed in Part 3)
This is generally more robust if you have enough data and want the model to learn specific patterns from your target shows/movies.

---
## Part 3: Fine-Tuning the Models with Your Data 🛠️

Your dataset consists of 20 samples of 5-minute audio with transcripts for 3 movies and 6 TV shows (9 distinct titles) in 6 different languages. This amounts to 100 minutes of total audio.

This is a **relatively small dataset** for fine-tuning, especially when spread across 9 titles and 6 languages. Fine-tuning might offer some adaptation, but substantial improvements over well-pre-trained base models will be challenging. However, it's a valuable learning experience.

### A. Fine-tuning `whisper.cpp`

**Goal**: To adapt the STT model to the specific accents, vocabulary, or acoustic environments present in your audio samples.

**Data Preparation**:
* **Segmentation**: Your 5-minute audio files should ideally be segmented into smaller chunks (e.g., 5-30 seconds) that align with the transcripts. Whisper processes audio in 30-second windows.
* **Accuracy**: Ensure your provided transcripts are highly accurate for these segments.
* **Format**: Prepare your data in a format suitable for Whisper fine-tuning scripts. This usually involves creating a dataset where each entry links an audio segment to its transcription and language. A common format is a CSV or JSONL file listing (`audio_file_path`, `transcription_text`, `language_code`).
    *Example: (`./audio_clips/show_A_lang1_seg1.wav`, "This is the dialogue.", "lang1_code")*

**Process**:
1.  Fine-tuning Whisper models is typically done in Python using libraries like Hugging Face `transformers`.
2.  You would load a pre-trained multilingual Whisper model (e.g., `openai/whisper-base` or `openai/whisper-small`).
3.  You'd then fine-tune it on your prepared multilingual dataset. The `transformers` library supports this.
4.  After fine-tuning in Python (which produces PyTorch checkpoints), you'll need to convert the fine-tuned model to the GGUF format that `whisper.cpp` uses. Scripts for this conversion are usually available within the `whisper.cpp` GitHub repository.

**Expected Outcome with ~100 minutes (avg. <17 mins/language)**:
* You might see minor improvements if your audio has very distinct characteristics (e.g., specific recurring names, consistent background noise) that the model can learn.
* Don't expect a drastic reduction in Word Error Rate (WER) across the board.
* It's crucial to have a small validation set (a portion of your data not used for training) to check if fine-tuning helps or overfits.

### B. Fine-tuning the LLM (for classification)

**Goal**: To teach an LLM to associate dialogue snippets (transcripts) with the correct movie or TV show title.

**Data Preparation**:
* **Input**: The transcripts from your dataset.
* **Labels**: The correct movie/TV show title for each transcript.
* **Format**: Create a dataset of (`text`, `label`) pairs. For example, a JSONL file:
    ```json
    {"text": "transcript from movie A, language X...", "label": "Movie A Title"}
    {"text": "transcript from TV show B, language Y...", "label": "TV Show B Title"}
    ...
    ```
* Since your transcripts are in 6 languages, your fine-tuning dataset will be multilingual. You'll need a base LLM with decent multilingual understanding.

**Process**:
1.  Select a base LLM that is compatible with `llama.cpp` (i.e., can be converted to GGUF). Smaller models like Llama 2 7B, Mistral 7B, or specialized smaller multilingual models are good candidates. Quantized versions are essential for CPU.
2.  Fine-tuning is also typically done in Python using frameworks like Hugging Face `transformers` and techniques like **LoRA (Low-Rank Adaptation)**. LoRA is highly recommended for small datasets as it's parameter-efficient and helps prevent catastrophic forgetting.
3.  You'll fine-tune the chosen LLM on your (`transcript`, `show_label`) pairs for text classification.
4.  After fine-tuning, convert the fine-tuned LLM (especially LoRA adapters if used) to the GGUF format for use with `llama.cpp`.

**Expected Outcome with ~100 minutes of transcripts (avg. ~11 mins/show)**:
* This is a **very small amount of data per class** (9 shows).
* Fine-tuning might help the model pick up on very distinctive phrases or character names if they consistently appear.
* There's a **high risk of overfitting**.
* Few-shot prompting with a good general-purpose LLM might be competitive or even outperform fine-tuning with such limited data. It's worth comparing both.
* Ensure your validation set for the LLM is representative.

---
## Part 4: C++ Project Integration 🧩

Your C++ application will need to tie these components together:

1.  **Audio Capture Module**: Uses PortAudio/RtAudio to get raw audio data.
2.  **STT Module (`whisper.cpp`)**:
    * Initializes `whisper.cpp` with the chosen (potentially fine-tuned) GGUF model.
    * Feeds captured audio to `whisper.cpp`.
    * Retrieves the transcribed text and detected language.
3.  **LLM Classification Module (`llama.cpp`)**:
    * Initializes `llama.cpp` with the chosen (potentially fine-tuned) GGUF LLM.
    * Creates a prompt (either for few-shot or just the transcript for a fine-tuned classification model).
    * Sends the transcript (and prompt) to `llama.cpp`.
    * Retrieves the classification result (the movie/TV show title).
4.  **Main Application Logic**:
    * Orchestrates the flow from audio input to final classification output.
    * Displays or logs the identified show.

### Key C++ aspects:

* Managing buffers for audio data.
* Interfacing with `whisper.cpp`'s C API (or C++ wrapper if you build one).
* Interfacing with `llama.cpp`'s C API (or C++ wrapper).
* **Multithreading** might be beneficial: one thread for audio capture, another for `whisper.cpp` processing, and another for `llama.cpp` processing to keep the pipeline responsive, though this adds complexity.

---
## Additional Considerations & Challenges ⚠️

* **Real-time Performance**: Achieving true real-time performance ("active listening" and instant classification) on a CPU will depend on the CPU's power, the size of the Whisper model, and the size of the LLM. You'll need to use smaller, heavily quantized models. There will be latency.
* **Accuracy**: Smaller models trade accuracy for speed. The STT accuracy directly impacts the LLM's ability to classify correctly.
* **Model Management**: Handling the GGUF model files and loading them efficiently.
* **Language Handling**: Ensure your entire pipeline correctly passes and uses language information if needed (e.g., if your LLM is language-specific, though a multilingual one is better).
* **Noise Robustness**: Real-world audio can be noisy. Whisper is generally robust, but extreme noise will degrade performance.
* **Computational Cost**: Even with `whisper.cpp` and `llama.cpp`, continuous STT and LLM inference can be CPU-intensive. Monitor system load.
