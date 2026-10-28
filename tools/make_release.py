"""Run from the repo root:  python tools/make_release.py
Trains both models, exports NumPy weights for the server, and builds the public handouts."""
import os, sys, shutil, zipfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
os.chdir(ROOT)
from setup_challenge import ComponentClassifier, train_model  # noqa: E402

DATA, DL = ROOT / "data", ROOT / "public" / "downloads"
DATA.mkdir(exist_ok=True)
DL.mkdir(parents=True, exist_ok=True)

train_model(str(DATA / "w1.pth"), seed=42, shift=False)
train_model(str(DATA / "w2.pth"), seed=43, shift=True)
shutil.move("resistor.png", DATA / "resistor.png")  # written by setup_challenge

for n in (1, 2):  # NumPy weights for the server (no torch on Vercel)
    m = ComponentClassifier()
    m.load_state_dict(torch.load(DATA / f"w{n}.pth", map_location="cpu"))
    np.savez(DATA / f"w{n}.npz", **{k: v.numpy() for k, v in m.state_dict().items()})

# public handouts: stage 1 only. w2.pth is NEVER copied here.
shutil.copy(DATA / "resistor.png", DL / "resistor.png")
shutil.copy(DATA / "w1.pth", DL / "w1.pth")
shutil.copy(ROOT / "tools/handout/model_def.py", DL / "model_def.py")
with zipfile.ZipFile(DL / "bouncer_stage1.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for f in ("resistor.png", "w1.pth", "model_def.py"):
        z.write(DL / f, f)
    z.write(ROOT / "tools/handout/README.txt", "README.txt")
print("[+] release built: data/ and public/downloads/ are ready")
