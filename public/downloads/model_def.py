import torch, torch.nn as nn, torch.nn.functional as F

CLASSES = ["Resistor", "Capacitor", "Inductor"]


class ComponentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten())
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 64), nn.ReLU(), nn.Linear(64, 3))

    def forward(self, x):
        return self.classifier(self.features(x))


def hardware_sensor_pipeline(x):      # x: (N,3,64,64), values in [0,1]
    q = torch.round(x * 15.0) / 15.0  # 4-bit ADC
    return F.avg_pool2d(q, kernel_size=3, stride=1, padding=1)
