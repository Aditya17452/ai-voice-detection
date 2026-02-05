import requests
import uuid

def download_audio_from_url(url: str) -> str:
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    filename = f"temp_{uuid.uuid4().hex}.mp3"
    with open(filename, "wb") as f:
        f.write(response.content)

    return filename
