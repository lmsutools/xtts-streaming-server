# XTTS Streaming Server - Easy Setup Guide

This guide provides a simplified explanation of how to set up and use the XTTS Streaming Server, even if you're a junior developer. Follow the steps below to get started:

## 1. Prerequisites

- Make sure you have Docker installed on your machine. If you don't have it yet, visit the official Docker website and follow their installation instructions for your operating system.

## 2. Run the Server

To run the XTTS Streaming Server, open a terminal and execute the following command:

```bash
$ docker run --gpus=all -e COQUI_TOS_AGREED=1 --rm -p 6006:80 ghcr.io/coqui-ai/xtts-streaming-server:latest-cuda121
```

This command will download the pre-built Docker image and start the server. The server will be accessible at `http://localhost:6006`.

Note: By setting the `COQUI_TOS_AGREED` environment variable to `1`, you agree to the terms of the [CPML license](https://coqui.ai/cpml).

## 3. Test the Server

Once the server is running, you can test it using the provided demo script. Follow these steps:

1. Clone the `xtts-streaming-server` repository:

```bash
$ git clone https://github.com/coqui-ai/xtts-streaming-server.git
```

2. Navigate to the `xtts-streaming-server` directory:

```bash
$ cd xtts-streaming-server
```

3. Install the required dependencies:

```bash
$ python -m pip install -r test/requirements.txt
```

4. Run the demo script:

```bash
$ python demo.py
```

This will launch a web-based demo application that allows you to interact with the XTTS Streaming Server.

## 4. Explore the Demo

In the demo application, you can:

- Select a pre-trained studio speaker or clone a new speaker by uploading a reference audio file.
- Enter the desired text and choose the language.
- Click the "TTS" button to generate audio using the selected speaker and text.
- Listen to the generated audio or download it.

That's it! You now have a basic understanding of how to set up and use the XTTS Streaming Server. Feel free to explore the code and experiment with different settings to further familiarize yourself with the project.

If you encounter any issues or have questions, refer to the original README file or reach out to the project maintainers for assistance.