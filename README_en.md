# 🔍 SubImageLocator

[中文](README_zh.md) | **English**

Fast sub-image localization powered by **SuperPoint + LightGlue** feature matching. Find where a sub-image appears in a larger image in ~28ms.

## ✨ Features

- **Fast**: ~28ms inference with model caching (NVIDIA L40 GPU)
- **Precise**: Returns bounding box as percentage coordinates + pixel coordinates
- **Visual**: Match region overlay, feature point visualization
- **WebUI**: Ready-to-use Gradio interface

## 📸 Capabilities

| Feature | Description |
|---------|-------------|
| Sub-image localization | Output x_min%, y_min%, x_max%, y_max% |
| Match visualization | Green bounding box on original image |
| Feature matching | Side-by-side keypoint correspondence |
| Confidence scoring | Based on RANSAC inlier ratio |

## 🚀 Quick Start

### Requirements

- Python 3.10+
- PyTorch (CUDA)

### Installation

```bash
git clone https://github.com/jx1100370217/SubImageLocator.git
cd SubImageLocator
pip install -r requirements.txt
pip install git+https://github.com/cvg/LightGlue.git
```

### Launch

```bash
bash start.sh
# Open http://localhost:7861
```

## 📁 Project Structure

```
SubImageLocator/
├── app.py                    # Gradio WebUI entry point
├── matchers/
│   ├── template_matcher.py   # MatchResult dataclass
│   └── feature_matcher.py    # SuperPoint+LightGlue matching
├── utils/
│   └── viz.py                # Visualization utilities
├── gen_examples.py           # Example image generator
├── datasets/examples/        # Example images
├── start.sh                  # Launch script (port 7861)
└── requirements.txt
```

## ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| Confidence threshold | 0.3 | RANSAC inlier ratio threshold |
| Min feature matches | 8 | Below this count = no match |

## 📊 Performance

| Stage | Time |
|-------|------|
| SuperPoint extraction | ~10ms |
| LightGlue matching | ~15ms |
| Homography + visualization | ~5ms |
| **Total (cached)** | **~30ms** |

> First call loads models to GPU (~2-3s). Subsequent calls reuse cache.

## 🔧 How It Works

1. **SuperPoint** extracts sparse keypoints (up to 2048) from both images
2. **LightGlue** matches features between the two images
3. **RANSAC + Homography** computes the projective transform
4. Template corners are projected onto the original image to get the bounding box

## License

MIT
