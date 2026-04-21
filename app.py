import os
import io
import base64
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# ── Model definition (mirrors training architecture) ──────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, residual=False):
        super().__init__()
        self.residual = residual and (in_ch == out_ch)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.GELU(),
        )
    def forward(self, x):
        out = self.block(x)
        return out + x if self.residual else out

class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
    def forward(self, x):
        avg = x.mean(1, keepdim=True)
        mx, _ = x.max(1, keepdim=True)
        return x * torch.sigmoid(self.conv(torch.cat([avg, mx], 1)))

class EuroSATNet(nn.Module):
    def __init__(self, num_classes=10, drop=0.4):
        super().__init__()
        self.net = nn.Sequential(
            # stage 1 — 64×64 → 32×32
            ConvBlock(3,   64),
            ConvBlock(64,  64,  residual=True),
            nn.MaxPool2d(2),
            # stage 2 — 32×32 → 16×16
            ConvBlock(64,  128),
            ConvBlock(128, 128, residual=True),
            nn.MaxPool2d(2),
            # stage 3 — 16×16 → 8×8
            ConvBlock(128, 256),
            ConvBlock(256, 256, residual=True),
            ConvBlock(256, 256, residual=True),
            SpatialAttention(),
            nn.MaxPool2d(2),
            # stage 4 — 8×8 → 4×4
            ConvBlock(256, 512),
            ConvBlock(512, 512, residual=True),
            SpatialAttention(),
            nn.MaxPool2d(2),
            # head
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(256, num_classes),
        )
    def forward(self, x):
        return self.net(x)

EuroSatCNN = EuroSATNet  # keep name consistent with checkpoint loading below


# ── Load checkpoint ───────────────────────────────────────────────────────────

MODEL_PATH = os.environ.get("MODEL_PATH", "model.pth")

checkpoint = torch.load(MODEL_PATH, map_location="cpu")
CLASS_NAMES  = checkpoint["class_names"]
NUM_CLASSES  = checkpoint["num_classes"]
IMG_SIZE     = checkpoint["img_size"]
MEAN         = checkpoint["mean"]
STD          = checkpoint["std"]
VAL_ACC      = checkpoint.get("val_acc", None)
EPOCHS       = checkpoint.get("epochs_trained", None)

model = EuroSATNet(num_classes=NUM_CLASSES)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ── Class metadata ────────────────────────────────────────────────────────────

CLASS_META = {
    "AnnualCrop":            {"icon": "🌾", "color": "#e8c547", "desc": "Seasonally harvested crops like wheat, corn, and sunflowers."},
    "Forest":                {"icon": "🌲", "color": "#2d7a3a", "desc": "Dense tree coverage including deciduous and coniferous forests."},
    "HerbaceousVegetation":  {"icon": "🌿", "color": "#6ab04c", "desc": "Low-growing non-woody plants, grasslands, and meadows."},
    "Highway":               {"icon": "🛣️",  "color": "#7f8c8d", "desc": "Major road infrastructure including motorways and expressways."},
    "Industrial":            {"icon": "🏭", "color": "#c0392b", "desc": "Manufacturing plants, warehouses, and industrial complexes."},
    "Pasture":               {"icon": "🐄", "color": "#a8e063", "desc": "Managed grasslands used for livestock grazing."},
    "PermanentCrop":         {"icon": "🍇", "color": "#8e44ad", "desc": "Long-cycle crops like vineyards, orchards, and olive groves."},
    "Residential":           {"icon": "🏘️",  "color": "#e67e22", "desc": "Housing areas including urban and suburban settlements."},
    "River":                 {"icon": "🏞️",  "color": "#3498db", "desc": "Flowing water bodies, streams, and riverbanks."},
    "SeaLake":               {"icon": "🌊", "color": "#1a6fa8", "desc": "Standing water bodies including seas, lakes, and reservoirs."},
}


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template(
        "index.html",
        class_names=CLASS_NAMES,
        class_meta=CLASS_META,
        val_acc=round(VAL_ACC * 100, 2) if VAL_ACC else None,
        epochs=EPOCHS,
        img_size=IMG_SIZE,
        num_classes=NUM_CLASSES,
    )


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    try:
        img_bytes = file.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

        # thumbnail for display
        thumb = img.copy()
        thumb.thumbnail((300, 300))
        buf = io.BytesIO()
        thumb.save(buf, format="PNG")
        img_b64 = base64.b64encode(buf.getvalue()).decode()

        tensor = transform(img).unsqueeze(0)

        with torch.no_grad():
            logits = model(tensor)
            probs  = F.softmax(logits, dim=1)[0].tolist()

        top_idx  = int(torch.argmax(torch.tensor(probs)))
        top_name = CLASS_NAMES[top_idx]
        top_conf = probs[top_idx]

        results = [
            {
                "class":      CLASS_NAMES[i],
                "confidence": round(probs[i] * 100, 2),
                "color":      CLASS_META[CLASS_NAMES[i]]["color"],
                "icon":       CLASS_META[CLASS_NAMES[i]]["icon"],
                "desc":       CLASS_META[CLASS_NAMES[i]]["desc"],
            }
            for i in range(NUM_CLASSES)
        ]
        results.sort(key=lambda x: x["confidence"], reverse=True)

        return jsonify({
            "top_class":      top_name,
            "top_confidence": round(top_conf * 100, 2),
            "top_icon":       CLASS_META[top_name]["icon"],
            "top_color":      CLASS_META[top_name]["color"],
            "top_desc":       CLASS_META[top_name]["desc"],
            "all_results":    results,
            "image_b64":      img_b64,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/model_info")
def model_info():
    return jsonify({
        "num_classes":  NUM_CLASSES,
        "class_names":  CLASS_NAMES,
        "img_size":     IMG_SIZE,
        "val_acc":      round(VAL_ACC * 100, 2) if VAL_ACC else None,
        "epochs":       EPOCHS,
        "mean":         MEAN,
        "std":          STD,
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)