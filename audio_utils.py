import base64
import uuid
from fastapi import HTTPException

def save_base64_mp3(base64_str: str) -> str:
    try:
        audio_bytes = base64.b64decode(base64_str)
    except Exception:
        raise HTTPException(
            status_code=400,
            detail="Invalid Base64 audio string"
        )

    filename = f"temp_{uuid.uuid4().hex}.mp3"
    with open(filename, "wb") as f:
        f.write(audio_bytes)

    return filename
