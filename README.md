# ShowClassifier

ShowClassifier is a C++ project for real-time audio capture and transcription, leveraging [whisper.cpp](whisper.cpp/) and custom audio processing components. This branch integrates model conversion scripts, audio capture utilities, and supports various Whisper model formats for speech-to-text tasks.

## Features

- **Real-time audio capture** using custom C++ modules.
- **Speech-to-text transcription** powered by [whisper.cpp](whisper.cpp/).
- **Model conversion scripts** for handling different Whisper model formats ([models/](models/)).
- **Support for quantized and fine-tuned models**.
- **CMake-based build system** for easy compilation and dependency management.

## Building the Project

1. **Clone the repository** and its submodules:
    ```sh
    git clone --recurse-submodules https://github.com/kamal-nabhan/ShowClassifier.git
    ```

2. **Clone Whisper.cpp** :
    ```sh
    git clone --recurse-submodules https://github.com/ggml-org/whisper.cpp.git
    ```

2. **Configure and build** using CMake:
    ```sh
    mkdir -p build
    cd build
    cmake ..
    make
    ```

3. The main executable (e.g., `ShowClassifier`) will be available in the `build/` directory.

## Usage

1. **Prepare a Whisper model**  
   Download or convert a model using the scripts in [models/](models/). See [models/README.md](models/README.md) and [whisper.cpp/models/README.md](whisper.cpp/models/README.md) for details.

2. **Run the application**  
   Example:
   ```sh
   ./build/ShowClassifier - /path/to/model.bin <API_Key>