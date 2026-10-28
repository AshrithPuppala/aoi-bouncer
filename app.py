import io, os, hmac, hashlib
from pathlib import Path
import numpy as np
from flask import Flask, request, jsonify, render_template_string, send_file
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view as swv

BASE = Path(__file__).parent
DATA = BASE / "data"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024
FLAG = os.environ["FLAG"]
SECRET = os.environ["TOKEN_SECRET"].encode()
CLASSES = ["Resistor", "Capacitor", "Inductor"]

STAGES = {
    1: dict(weights="w1.npz", eps=36 / 255, randomized=False, hints=True),
    2: dict(weights="w2.npz", eps=28 / 255, randomized=True, hints=False),
}
for s in STAGES.values():
    s["W"] = dict(np.load(DATA / s["weights"]))

orig_img = np.asarray(Image.open(DATA / "resistor.png").convert("RGB"),
                      dtype=np.float32).transpose(2, 0, 1) / 255.0
MASK = np.zeros_like(orig_img)
MASK[:, 22:43, 16:49] = 1.0  # inclusive body bounds
rng = np.random.default_rng()


# ---- stateless stage gating (HMAC token) ----
def stage2_token():
    return hmac.new(SECRET, b"stage2", hashlib.sha256).hexdigest()[:24]


def token_ok(tok):
    return hmac.compare_digest(tok or "", stage2_token())


# ---- NumPy model + sensor ----
def conv3(x, w, b):
    xp = np.pad(x, ((0, 0), (1, 1), (1, 1)))
    return np.einsum("chwij,ocij->ohw", swv(xp, (3, 3), axis=(1, 2)), w) + b[:, None, None]


def pool2(x):
    c, h, w = x.shape
    return x.reshape(c, h // 2, 2, w // 2, 2).max(axis=(2, 4))


def model(x, W):
    x = pool2(np.maximum(conv3(x, W["features.0.weight"], W["features.0.bias"]), 0))
    x = pool2(np.maximum(conv3(x, W["features.3.weight"], W["features.3.bias"]), 0))
    x = np.maximum(W["classifier.0.weight"] @ x.reshape(-1) + W["classifier.0.bias"], 0)
    return W["classifier.2.weight"] @ x + W["classifier.2.bias"]


def sensor(x):
    q = np.round(x * np.float32(15.0)) / np.float32(15.0)
    qp = np.pad(q, ((0, 0), (1, 1), (1, 1)))
    return swv(qp, (3, 3), axis=(1, 2)).sum(axis=(3, 4)) / np.float32(9.0)


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def predict(x, cfg):
    W = cfg["W"]
    if not cfg["randomized"]:
        p = softmax(model(sensor(x), W))
        return p, int(p.argmax()), None
    ps, votes = [], np.zeros(3, int)
    for _ in range(7):
        dy, dx = rng.integers(-1, 2, size=2)
        xs = np.roll(x, (int(dy), int(dx)), axis=(1, 2))
        xs = np.clip(xs + rng.normal(0, 0.02, xs.shape).astype(np.float32), 0, 1)
        p = softmax(model(sensor(xs), W))
        ps.append(p)
        votes[int(p.argmax())] += 1
    return np.mean(ps, axis=0), int(votes.argmax()), votes


HTML_UI = """
<!DOCTYPE html><html><head><title>AOI Bouncer</title>
<style>
 body{background:#121212;color:#00ff9d;font-family:monospace;text-align:center;padding:40px}
 .box{border:1px solid #00ff9d;padding:25px;max-width:560px;margin:20px auto;background:#1e1e1e}
 input,button{margin-top:12px;padding:10px;background:#00ff9d;color:#121212;border:none;font-weight:bold}
 a{color:#00ff9d}
</style></head><body>
<div class="box">
 <h2>[AOI v2.4 - CAPACITOR GATEKEEPER]</h2>
 <p>Only verified <b>Capacitors</b> are permitted into the SMT assembly line.</p>
 <p><a href="/downloads/bouncer_stage1.zip">Download Stage 1 files</a> |
    <a href="/downloads/resistor.png">resistor.png</a></p>
</div>
<div class="box"><h3>Stage 1</h3>
 <form action="/verify/1" method="post" enctype="multipart/form-data">
  <input type="file" name="file" accept="image/png" required><br>
  <button>SCAN COMPONENT</button></form></div>
<div class="box"><h3>Stage 2</h3>
 <form action="/verify/2" method="post" enctype="multipart/form-data">
  <input type="text" name="token" placeholder="stage token" required><br>
  <input type="file" name="file" accept="image/png" required><br>
  <button>FINAL INSPECTION</button></form></div>
</body></html>
"""


@app.route("/")
def index():
    return render_template_string(HTML_UI)


@app.route("/stage2/model")  # download unlocked by the stage 1 token
def stage2_model():
    if not token_ok(request.args.get("token")):
        return jsonify({"error": "Invalid token"}), 403
    return send_file(DATA / "w2.pth", as_attachment=True)


@app.route("/verify", methods=["POST"])
def verify_default():
    return handle(1)


@app.route("/verify/<int:stage>", methods=["POST"])
def verify(stage):
    return handle(stage)


def handle(stage):
    cfg = STAGES.get(stage)
    if not cfg:
        return jsonify({"error": "Unknown stage"}), 404
    if stage == 2 and not token_ok(request.form.get("token")):
        return jsonify({"error": "Missing or invalid stage token"}), 403

    def deny(status, msg):
        body = {"status": status}
        if cfg["hints"]:
            body["message"] = msg
        return jsonify(body), 403

    f = request.files.get("file")
    if not f:
        return jsonify({"error": "No file uploaded"}), 400
    try:
        img = Image.open(io.BytesIO(f.read())).convert("RGB")
    except Exception:
        return jsonify({"error": "Invalid image"}), 400
    if img.size != (64, 64):
        return jsonify({"error": "Resolution mismatch! Sensor requires 64x64 PNG."}), 400

    x = np.asarray(img, dtype=np.float32).transpose(2, 0, 1) / 255.0
    if np.abs((x - orig_img) * (1.0 - MASK)).max() > 1e-4:
        return deny("TAMPER_DETECTED",
                    "Background PCB or leads modified! Perturbations restricted to component body.")
    linf = float(np.abs(x - orig_img).max())
    if linf > cfg["eps"]:
        return deny("VISUAL_ANOMALY",
                    f"L-inf perturbation {linf:.3f} exceeds optical tolerance ({cfg['eps']:.3f}).")

    p, pred, votes = predict(x, cfg)
    ok = pred == 1 and p[1] > 0.80 and (votes is None or votes[1] >= 6)

    result = {"detected_component": CLASSES[pred],
              "confidence": {c: round(float(v), 4) for c, v in zip(CLASSES, p)},
              "max_pixel_delta": round(linf, 4)}
    if votes is not None:
        result["votes"] = {c: int(v) for c, v in zip(CLASSES, votes)}

    if ok and stage == 1:
        result["token"] = stage2_token()
        result["message"] = ("STAGE 1 CLEARED. Final inspection unlocked: "
                             "GET /stage2/model?token=<token>, then POST /verify/2.")
    elif ok:
        result["flag"] = FLAG
        result["message"] = "ACCESS GRANTED: Capacitor verified!"
    else:
        result["message"] = "ACCESS DENIED: Component rejected."
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
