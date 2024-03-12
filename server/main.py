import base64
import io
import os
import tempfile
import wave
import torch
import numpy as np
from typing import List
from pydantic import BaseModel
import uvicorn
import traceback
import asyncio
import concurrent.futures

from fastapi import FastAPI, UploadFile, Body, Request
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager

torch.set_num_threads(int(os.environ.get("NUM_THREADS", os.cpu_count())))
device = torch.device("cuda" if os.environ.get("USE_CPU", "0") == "0" else "cpu")
if not torch.cuda.is_available() and device == "cuda":
    raise RuntimeError("CUDA device unavailable, please use Dockerfile.cpu instead.")

custom_model_path = os.environ.get("CUSTOM_MODEL_PATH", "/app/tts_models")



async def load_model():
    loop = asyncio.get_running_loop()

    if os.path.exists(custom_model_path) and os.path.isfile(custom_model_path + "/config.json"):
        model_path = custom_model_path
        print("Loading custom model from", model_path, flush=True)
    else:
        print("Loading default model", flush=True)
        model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
        print("Downloading XTTS Model:", model_name, flush=True)

        async def download_model_async(model_name):
            model_manager = ModelManager()
            await loop.run_in_executor(None, model_manager.download_model, model_name)

        await download_model_async(model_name)
        model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
        print("XTTS Model downloaded", flush=True)

    print("Loading XTTS", flush=True)
    try:
        config = XttsConfig()
        await loop.run_in_executor(None, config.load_json, os.path.join(model_path, "config.json"))
        model = Xtts.init_from_config(config)

        def load_checkpoint():
            model.load_checkpoint(config, checkpoint_dir=model_path, eval=True, use_deepspeed=True if device == "cuda" else False)

        await loop.run_in_executor(None, load_checkpoint)

        def move_to_device():
            model.to(device)

        await loop.run_in_executor(None, move_to_device)

        print("XTTS Loaded.", flush=True)
        return model
    except Exception as e:
        print(f"Error loading XTTS: {str(e)}", flush=True)
        traceback.print_exc()
        exit(1)

model = asyncio.run(load_model())

def postprocess(wav):
    """Post process the output waveform"""
    if isinstance(wav, list):
        wav = torch.cat(wav, dim=0)
    wav = wav.clone().detach().cpu().numpy()
    wav = wav[None, : int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav

def encode_audio_common(
    frame_input, encode_base64=True, sample_rate=24000, sample_width=2, channels=1
):
    """Return base64 encoded audio"""
    wav_buf = io.BytesIO()
    with wave.open(wav_buf, "wb") as vfout:
        vfout.setnchannels(channels)
        vfout.setsampwidth(sample_width)
        vfout.setframerate(sample_rate)
        vfout.writeframes(frame_input)

    wav_buf.seek(0)
    if encode_base64:
        b64_encoded = base64.b64encode(wav_buf.getbuffer()).decode("utf-8")
        return b64_encoded
    else:
        return wav_buf.read()

class StreamingInputs(BaseModel):
    speaker_embedding: List[float]
    gpt_cond_latent: List[List[float]]
    text: str
    language: str
    add_wav_header: bool = True
    stream_chunk_size: str = "20"

class TTSInputs(BaseModel):
    speaker_embedding: List[float]
    gpt_cond_latent: List[List[float]]
    text: str
    language: str

def create_app():
    app = FastAPI(
        title="XTTS Streaming server",
        description="""XTTS Streaming server""",
        version="0.0.1",
        docs_url="/",
    )

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.post("/clone_speaker")
    async def predict_speaker(wav_file: UploadFile):
        """Compute conditioning inputs from reference audio file."""
        temp_audio_name = next(tempfile._get_candidate_names())
        with open(temp_audio_name, "wb") as temp:
            temp.write(io.BytesIO(await wav_file.read()).getbuffer())
            loop = asyncio.get_running_loop()

            def get_conditioning_latents():
                return model.get_conditioning_latents(temp_audio_name)

            gpt_cond_latent, speaker_embedding = await loop.run_in_executor(None, get_conditioning_latents)
        return {
            "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().half().tolist(),
            "speaker_embedding": speaker_embedding.cpu().squeeze().half().tolist(),
        }

    async def predict_streaming_generator(parsed_input: dict = Body(...)):
        speaker_embedding = torch.tensor(parsed_input.speaker_embedding).unsqueeze(0).unsqueeze(-1).to(device, non_blocking=True)
        gpt_cond_latent = torch.tensor(parsed_input.gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0).to(device, non_blocking=True)
        text = parsed_input.text
        language = parsed_input.language

        stream_chunk_size = int(parsed_input.stream_chunk_size)
        add_wav_header = parsed_input.add_wav_header

        async for chunk in model.inference_stream(
            text,
            language,
            gpt_cond_latent,
            speaker_embedding,
            stream_chunk_size=stream_chunk_size,
            enable_text_splitting=True
        ):
            chunk = postprocess(chunk)
            if add_wav_header:
                yield encode_audio_common(b"", encode_base64=False)
                add_wav_header = False
            yield chunk.tobytes()

    @app.post("/tts_stream", response_class=StreamingResponse)
    async def predict_streaming_endpoint(parsed_input: StreamingInputs):
        return StreamingResponse(
            predict_streaming_generator(parsed_input.dict()),
            media_type="audio/wav",
        )

    @app.post("/tts")
    async def predict_speech(parsed_input: TTSInputs):
        speaker_embedding = torch.tensor(parsed_input.speaker_embedding).unsqueeze(0).unsqueeze(-1).to(device, non_blocking=True)
        gpt_cond_latent = torch.tensor(parsed_input.gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0).to(device, non_blocking=True)
        text = parsed_input.text
        language = parsed_input.language

        loop = asyncio.get_running_loop()

        def inference_call():
            return model.inference(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
            )

        out = await loop.run_in_executor(None, inference_call)

        wav = postprocess(torch.tensor(out["wav"]))

        return encode_audio_common(wav.tobytes())

    @app.get("/studio_speakers")
    async def get_speakers():
        if hasattr(model, "speaker_manager") and hasattr(model.speaker_manager, "speakers"):
            return {
                speaker: {
                    "speaker_embedding": model.speaker_manager.speakers[speaker]["speaker_embedding"].cpu().squeeze().half().tolist(),
                    "gpt_cond_latent": model.speaker_manager.speakers[speaker]["gpt_cond_latent"].cpu().squeeze().half().tolist(),
                }
                for speaker in model.speaker_manager.speakers.keys()
            }
        else:
            return {}

    @app.get("/languages")
    async def get_languages():
        return config.languages

    # Add a catch-all route to handle 404 Not Found errors
    @app.exception_handler(404)
    async def not_found_handler(request: Request, exc):
        return JSONResponse(
            status_code=404,
            content={"message": "The requested resource was not found."},
        )

    return app

app = create_app()

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006)