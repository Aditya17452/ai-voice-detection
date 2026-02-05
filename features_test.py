import librosa
import numpy as np

SAMPLE_RATE = 16000
N_MELS = 64
MAX_TIME = 300

def extract_mel_spectrogram(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    audio, _ = librosa.effects.trim(audio)

    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=N_MELS
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    if mel_db.shape[1] < MAX_TIME:
        mel_db = np.pad(
            mel_db,
            ((0, 0), (0, MAX_TIME - mel_db.shape[1]))
        )
    else:
        mel_db = mel_db[:, :MAX_TIME]

    mel_db = mel_db[np.newaxis, np.newaxis, :, :]
    return mel_db.astype("float32")
