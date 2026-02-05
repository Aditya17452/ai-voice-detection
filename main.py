from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from auth import verify_api_key
from audio_utils import save_base64_mp3
from inference import predict_audio
import os

app = FastAPI(title="AI Voice Detection API")

from typing import Optional
from pydantic import BaseModel, Field

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: Optional[str] = Field(default=None)
    audioUrl: Optional[str] = Field(default=None)


from audio_fetcher import download_audio_from_url
from audio_utils import save_base64_mp3

@app.post("/api/voice-detection")
def detect_voice(
    request: VoiceRequest,
    api_key: str = Depends(verify_api_key)
):
    if not request.audioBase64 and not request.audioUrl:
    raise HTTPException(
        status_code=400,
        detail="Either audioBase64 or audioUrl must be provided"
    )


    if request.audioFormat.lower() != "mp3":
        raise HTTPException(status_code=400, detail="Only mp3 supported")

    if request.audioBase64:
        audio_path = save_base64_mp3(request.audioBase64)

    elif request.audioUrl:
        audio_path = download_audio_from_url(request.audioUrl)

    else:
        raise HTTPException(
            status_code=400,
            detail="Either audioBase64 or audioUrl must be provided"
        )

    try:
        result = predict_audio(audio_path)
    finally:
        os.remove(audio_path)

    return {
        "status": "success",
        "language": request.language,
        "classification": result["classification"],
        "confidenceScore": result["confidenceScore"],
        "explanation": (
            "Unnatural pitch consistency detected"
            if result["classification"] == "AI_GENERATED"
            else "Natural human voice variations detected"
        )
    }



