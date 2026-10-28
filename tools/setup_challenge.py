import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image, ImageDraw
import torchvision.transforms as T  # noqa: F401  (kept for parity with original)


class ComponentClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64x64 -> 32x32
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32x32 -> 16x16
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 16 * 16, 64),
            nn.ReLU(),
            nn.Linear(64, 3),  # 0: Resistor, 1: Capacitor, 2: Inductor
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def create_base_resistor():
    img = Image.new("RGB", (64, 64), (20, 40, 20))  # dark green PCB
    draw = ImageDraw.Draw(img)
    draw.line([(4, 32), (60, 32)], fill=(190, 190, 190), width=3)       # leads
    draw.rectangle([16, 22, 48, 42], fill=(210, 180, 140))              # body (inclusive)
    draw.rectangle([20, 22, 23, 42], fill=(200, 30, 30))                # bands
    draw.rectangle([27, 22, 30, 42], fill=(140, 30, 180))
    draw.rectangle([34, 22, 37, 42], fill=(220, 200, 30))
    draw.rectangle([42, 22, 44, 42], fill=(212, 175, 55))
    img.save("resistor.png")  # written to the current working directory
    return T.ToTensor()(img)


def train_model(out, seed, shift):
    """shift=True adds +-1px roll augmentation so the model survives stage-2 jitter."""
    torch.manual_seed(seed)
    base_res = create_base_resistor()

    base_cap = base_res.clone()
    base_cap[:, 22:42, 16:48] = torch.tensor([0.1, 0.4, 0.85]).view(3, 1, 1)
    base_ind = base_res.clone()
    base_ind[:, 22:42, 16:48] = torch.tensor([0.75, 0.35, 0.15]).view(3, 1, 1)

    def aug(b):
        x = b + torch.randn_like(b) * 0.04
        if shift:
            dy, dx = torch.randint(-1, 2, (2,)).tolist()
            x = torch.roll(x, (dy, dx), (1, 2))
        return x.clamp(0, 1)

    model = ComponentClassifier()
    opt = optim.Adam(model.parameters(), lr=0.005)
    crit = nn.CrossEntropyLoss()
    for _ in range(150):
        xs, ys = [], []
        for _ in range(32):
            for b, label in ((base_res, 0), (base_cap, 1), (base_ind, 2)):
                xs.append(aug(b))
                ys.append(label)
        opt.zero_grad()
        crit(model(torch.stack(xs)), torch.tensor(ys)).backward()
        opt.step()
    torch.save(model.state_dict(), out)


if __name__ == "__main__":
    train_model("w1.pth", seed=42, shift=False)
    train_model("w2.pth", seed=43, shift=True)
    print("[+] w1.pth, w2.pth, resistor.png written")
