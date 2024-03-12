import asyncio
import base64
import io
import logging
import os
import tempfile
import wave
from typing import List

import aiofiles
import numpy as np
import torch
import uvicorn
from fastapi import Body, FastAPI, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set up device and threads
torch.set_num_threads(int(os.environ.get("NUM_THREADS", os.cpu_count())))
device = torch.device("cuda" if torch.cuda.is_available() and os.environ.get("USE_CPU", "0") == "0" else "cpu")

# Load the model
custom_model_path = os.environ.get("CUSTOM_MODEL_PATH", "/app/tts_models")

if os.path.exists(custom_model_path) and os.path.isfile(custom_model_path + "/config.json"):
    model_path = custom_model_path
    logger.info(f"Loading custom model from {model_path}")
else:
    logger.info("Loading default model")
    model_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    logger.info(f"Downloading XTTS Model: {model_name}")
    ModelManager().download_model(model_name)
    model_path = os.path.join(get_user_data_dir("tts"), model_name.replace("/", "--"))
    logger.info("XTTS Model downloaded")

logger.info("Loading XTTS")
config = XttsConfig()
config.load_json(os.path.join(model_path, "config.json"))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=model_path, eval=True, use_deepspeed=device.type == "cuda")
model.to(device)
logger.info("XTTS Loaded.")

# Set up FastAPI app
app = FastAPI(
    title="XTTS Streaming server",
    description="XTTS Streaming server",
    version="0.0.1",
    docs_url="/",
)

# Helper functions
def postprocess(wav):
    """Post process the output waveform"""
    if isinstance(wav, list):
        wav = torch.cat(wav, dim=0)
    wav = wav.clone().detach().cpu().numpy()
    wav = wav[None, : int(wav.shape[0])]
    wav = np.clip(wav, -1, 1)
    wav = (wav * 32767).astype(np.int16)
    return wav

def encode_audio_common(frame_input, encode_base64=True, sample_rate=24000, sample_width=2, channels=1):
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

# Pydantic models
class StreamingInputs(BaseModel):
    speaker_embedding: List[float]
    gpt_cond_latent: List[List[float]]
    text: str
    language: str
    add_wav_header: bool = True
    stream_chunk_size: int = 20

class TTSInputs(BaseModel):
    speaker_embedding: List[float]
    gpt_cond_latent: List[List[float]]
    text: str
    language: str

# API endpoints
@app.post("/clone_speaker")
async def predict_speaker(wav_file: UploadFile):
    temp_audio_name = next(tempfile._get_candidate_names())
    async with aiofiles.open(temp_audio_name, "wb") as temp:
        content = await wav_file.read()
        await temp.write(content)
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(temp_audio_name)
    return {
        "gpt_cond_latent": gpt_cond_latent.cpu().squeeze().half().tolist(),
        "speaker_embedding": speaker_embedding.cpu().squeeze().half().tolist(),
    }

async def predict_streaming_generator(parsed_input: StreamingInputs):
    speaker_embedding = torch.tensor(parsed_input.speaker_embedding).unsqueeze(0).unsqueeze(-1).to(device)
    gpt_cond_latent = torch.tensor(parsed_input.gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0).to(device)
    text = parsed_input.text
    language = parsed_input.language
    add_wav_header = parsed_input.add_wav_header
    stream_chunk_size = parsed_input.stream_chunk_size

    try:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            chunks = model.inference_stream(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
                stream_chunk_size=stream_chunk_size,
                enable_text_splitting=True
            )

            async for i, chunk in enumerate(chunks):
                chunk = postprocess(chunk)
                if i == 0 and add_wav_header:
                    yield encode_audio_common(b"", encode_base64=False)
                yield chunk.tobytes()
    except Exception as e:
        logger.exception("Error in predict_streaming_generator")
        raise HTTPException(status_code=500, detail="An error occurred during streaming") from e

@app.post("/tts_stream")
async def predict_streaming_endpoint(parsed_input: StreamingInputs):
    return StreamingResponse(
        predict_streaming_generator(parsed_input),
        media_type="audio/wav",
    )

@app.post("/tts")
async def predict_speech(parsed_input: TTSInputs):
    speaker_embedding = torch.tensor(parsed_input.speaker_embedding).unsqueeze(0).unsqueeze(-1).to(device)
    gpt_cond_latent = torch.tensor(parsed_input.gpt_cond_latent).reshape((-1, 1024)).unsqueeze(0).to(device)
    text = parsed_input.text
    language = parsed_input.language

    try:
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            out = model.inference(
                text,
                language,
                gpt_cond_latent,
                speaker_embedding,
            )

        wav = postprocess(torch.tensor(out["wav"]))
        return encode_audio_common(wav.tobytes())
    except Exception as e:
        logger.exception("Error in predict_speech")
        raise HTTPException(status_code=500, detail="An error occurred during TTS generation") from e

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=6006)