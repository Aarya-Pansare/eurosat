import io
import base64
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import streamlit as st

st.set_page_config(page_title="Land Cover Classifier", layout="centered")

# ── Model classes (same as yours) ─────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class SpatialAttn(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
    def forward(self, x):
        avg = x.mean(dim=1, keepdim=True)
        mx, _ = x.max(dim=1, keepdim=True)
        attn = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * attn

class EuroSatCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            ConvBlock(3, 64),
            ConvBlock(64, 64),
            nn.MaxPool2d(2, 2),
            ConvBlock(64, 128),
            ConvBlock(128, 128),
            nn.MaxPool2d(2, 2),
            ConvBlock(128, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 256),
            SpatialAttn(),
            nn.MaxPool2d(2, 2),
            ConvBlock(256, 512),
            ConvBlock(512, 512),
            SpatialAttn(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)

# ── Load model ────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    checkpoint = torch.load("model.pth", map_location="cpu")
    model = EuroSatCNN(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, checkpoint

model, checkpoint = load_model()

CLASS_NAMES = checkpoint["class_names"]
IMG_SIZE = checkpoint["img_size"]
MEAN = checkpoint["mean"]
STD = checkpoint["std"]

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=MEAN, std=STD),
])

# ── UI ────────────────────────────────────────────────────────────────────────

st.title("🌍 Land Cover Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0]

    top_idx = torch.argmax(probs).item()
    st.success(f"Prediction: {CLASS_NAMES[top_idx]} ({probs[top_idx]*100:.2f}%)")

    st.subheader("All Classes")
    for i, cls in enumerate(CLASS_NAMES):
        st.write(f"{cls}: {probs[i]*100:.2f}%")