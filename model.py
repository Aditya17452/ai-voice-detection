import torch
import torch.nn as nn

class CNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()

        self.cnn = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # Input size = 16 channels × 32 mel bins
        self.lstm = nn.LSTM(
            input_size=16 * 32,
            hidden_size=64,
            batch_first=True
        )

        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        # x: (B, 1, 64, 300)
        x = self.cnn(x)              # (B, 16, 32, 150)
        b, c, h, t = x.shape
        x = x.permute(0, 3, 1, 2)    # (B, 150, 16, 32)
        x = x.reshape(b, t, -1)      # (B, 150, 16*32)
        x, _ = self.lstm(x)
        x = x[:, -1, :]
        return self.fc(x)
