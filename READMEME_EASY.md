# XTTS Streaming Server

This repository contains a streaming server for the XTTS (Cross-lingual Text-to-Speech) model, which allows for real-time text-to-speech synthesis in multiple languages. The server utilizes the XTTS model to generate high-quality audio from given text input.

## Features

- Real-time text-to-speech synthesis
- Support for multiple languages
- Speaker cloning from reference audio
- Easy-to-use web-based demo application

## Prerequisites

- Python 3.7 or above
- PyTorch
- CUDA (if using GPU)
- Other dependencies listed in `server/requirements.txt` and `test/requirements.txt`

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/coqui-ai/xtts-streaming-server.git
   cd xtts-streaming-server


2. Create a new virtual environment:
   ```
   python -m venv venv
   Linux:
   source venv/bin/activate
   Windows:
   .\venv\Scripts\Activate.ps1
   ```

3. Install the required dependencies for the server:
   ```
 
   pip install -r server/requirements.txt
   ```

4. Download the XTTS model:
   - Run the `server/main.py` script once to download the default XTTS model:
     ```
     python server/main.py
     ```
   - The script will automatically download the model if it doesn't exist locally.

## Running the Server

1. Start the XTTS Streaming Server:
   ```
   python server/main.py
   ```
   The server will be accessible at `http://localhost:6006`.

2. (Optional) Customize the server:
   - If you want to use a custom XTTS model, place your model files in a directory and set the `CUSTOM_MODEL_PATH` environment variable to the path of that directory.
   - Adjust other server settings in the `server/main.py` script as needed.

## Testing the Server

1. Install the demo dependencies:
   ```
   pip install -r test/requirements.txt
   ```

2. Run the demo application:
   ```
   python demo.py
   ```
   This will start the demo application, and you can access it in your web browser at the provided URL.

3. Explore the demo:
   - Select a pre-trained studio speaker or clone a new speaker by uploading a reference audio file.
   - Enter the desired text and choose the language.
   - Click the "TTS" button to generate audio using the selected speaker and text.
   - Listen to the generated audio or download it.

## API Endpoints

The server exposes the following API endpoints:

- `/clone_speaker` (POST): Compute conditioning inputs from a reference audio file.
- `/tts_stream` (POST): Generate text-to-speech audio in real-time using the specified speaker and text.
- `/tts` (POST): Generate text-to-speech audio using the specified speaker and text.
- `/studio_speakers` (GET): Get the list of available pre-trained studio speakers.
- `/languages` (GET): Get the list of supported languages.

For detailed information on the request and response formats of each endpoint, please refer to the code in `server/main.py`.

## License

This project is licensed under the CPML License - see the [LICENSE](LICENSE) file for details. By running the code in this repository, you agree to the terms of the CPML license.

To indicate your agreement, set the environment variable `COQUI_TOS_AGREED` to `1`.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## Acknowledgements

- The XTTS model is developed by [Coqui](https://coqui.ai/).
- This project is built using [FastAPI](https://fastapi.tiangolo.com/) and [PyTorch](https://pytorch.org/).

## Contact

For any questions or inquiries, please contact the project maintainers or open an issue on the GitHub repository.