import torch
from model import CNNLSTM
from features_test import extract_mel_spectrogram

DEVICE = "cpu"   # force CPU for now (Windows safe)

model = CNNLSTM()
model.load_state_dict(torch.load("voice_model.pth", map_location="cpu"))
model.eval()

def predict_audio(file_path):
    mel = extract_mel_spectrogram(file_path)
    mel_tensor = torch.tensor(mel)

    with torch.no_grad():
        output = model(mel_tensor)
        probs = torch.softmax(output, dim=1)
        conf, pred = torch.max(probs, dim=1)

    return {
        "classification": "AI_GENERATED" if pred.item() == 1 else "HUMAN",
        "confidenceScore": round(conf.item(), 3)
    }
