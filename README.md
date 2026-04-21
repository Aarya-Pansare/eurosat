# TERRAVIEW — Sentinel-2 Land Cover Classification

A web application for classifying land cover from Sentinel-2 satellite imagery using your trained EuroSAT CNN model.

## Features
- Drag-and-drop image upload
- Instant inference with confidence scores for all 10 classes
- Visual probability bars per class
- Model metadata (accuracy, epochs, input size)
- 10 EuroSAT classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake

## Setup

### 1. Install dependencies
```bash
pip install flask torch torchvision pillow
```

### 2. Place your model
Copy your `.pth` file into this folder and rename it `model.pth`, **or** pass the path as an argument to `run.sh`.

### 3. Run
```bash
chmod +x run.sh
./run.sh                          # uses ./model.pth by default
./run.sh /path/to/your_model.pth  # custom path
```

Then open **http://localhost:5000** in your browser.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Web UI |
| `/predict` | POST | Classify an image (multipart/form-data, field: `file`) |
| `/model_info` | GET | Returns model metadata as JSON |

### Example `/predict` curl call
```bash
curl -X POST http://localhost:5000/predict \
  -F "file=@your_image.jpg"
```

## Model Info
- Architecture: Custom CNN with Spatial Attention
- Dataset: EuroSAT (Sentinel-2)
- Input size: 64×64 RGB
- Validation accuracy: ~98%
- Classes: 10
