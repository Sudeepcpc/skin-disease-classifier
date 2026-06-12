# 🔬 Skin Disease Classifier

> Deep learning model to detect 7 types of skin diseases from photos — built to help rural communities with limited access to dermatologists.

## 🚀 Live Demo
**[👉 Try it here — huggingface.co/spaces/sudeep1947/skin-disease-classifier](https://huggingface.co/spaces/sudeep1947/skin-disease-classifier)**

---

## 🎯 The Problem

India has 270 million farmers working outdoors in harsh sun every day. Most villages have zero dermatologists. Skin cancer detected late has very low survival rates. Detected early — 99% survival rate.

This project is a first-level screening tool that can flag dangerous skin conditions and tell people *"please see a doctor urgently."*

---

## 📊 Model Performance

| Metric | Score |
|---|---|
| Overall Accuracy | **77.4%** |
| RocAuc Score | **0.9628** |
| Training Images | 10,015 |
| Disease Classes | 7 |
| Architecture | ResNet50 (Transfer Learning) |

### Per-Disease Results

| Disease | Precision | Recall | F1 | Danger Level |
|---|---|---|---|---|
| 🚨 Melanoma (mel) | 0.49 | **0.71** | 0.58 | Cancer |
| 🚨 Basal Cell Carcinoma (bcc) | 0.70 | 0.81 | 0.75 | Cancer |
| ⚠️ Actinic Keratosis (akiec) | 0.60 | 0.48 | 0.53 | Pre-cancer |
| ✅ Melanocytic Nevi (nv) | 0.97 | 0.79 | 0.87 | Harmless |
| ✅ Benign Keratosis (bkl) | 0.55 | 0.78 | 0.65 | Harmless |
| ✅ Dermatofibroma (df) | 0.46 | 0.87 | 0.60 | Harmless |
| ✅ Vascular Lesion (vasc) | 0.72 | 0.97 | 0.83 | Harmless |

### Key Achievement — Melanoma Detection

```
Baseline model (no fix):   recall = 0.33  (missed 67% of cancers)
Final model (with fix):    recall = 0.71  (misses only 29% of cancers)

Improvement: +115% better melanoma detection
```

---

## 🏥 Diseases Detected

| Code | Full Name | Description |
|---|---|---|
| `mel` | Melanoma | Most dangerous skin cancer. Can spread to organs if undetected. |
| `bcc` | Basal Cell Carcinoma | Most common skin cancer. Slow growing, treatable if caught early. |
| `akiec` | Actinic Keratosis | Pre-cancer caused by sun exposure. High risk for outdoor workers. |
| `nv` | Melanocytic Nevi | Common moles. Harmless unless changing shape or color. |
| `bkl` | Benign Keratosis | Non-cancerous waxy skin growth. Very common in older people. |
| `df` | Dermatofibroma | Harmless small firm bump. Usually caused by minor injury. |
| `vasc` | Vascular Lesion | Blood vessel abnormality near skin. Mostly harmless. |

---

## 🧠 Technical Approach

### 1. Dataset — HAM10000
- 10,015 dermoscopy images across 7 classes
- Source: [Kaggle — kmader/skin-cancer-mnist-ham10000](https://www.kaggle.com/datasets/kmader/skin-lesion-analysis-toward-melanoma-detection)
- Split: 80% training / 20% validation

### 2. Class Imbalance Problem & Fix
The dataset is heavily imbalanced:
```
nv (moles):    6,705 images   ← 67% of dataset
df (rare):       115 images   ← 1% of dataset
```
Without fixing this, the model just predicts "mole" for everything.

**Fix:** Weighted loss function — rare diseases get proportionally higher weight during training.
```python
counts = df['dx'].value_counts()
weights = [1.0/counts[c] for c in dls.vocab]
class_weights = tensor(weights).float().cuda()
loss_func = CrossEntropyLossFlat(weight=class_weights)
```

### 3. Transfer Learning — ResNet50
- ResNet50 pretrained on ImageNet (already knows edges, shapes, textures)
- Fine-tuned on skin disease images
- freeze_epochs=3 → trained head layers first, then full network

### 4. Training
```
Architecture:  ResNet50
Epochs:        8 (+ 3 freeze epochs)
Batch size:    32
Image size:    224x224
Augmentation:  Random flips, rotations, zoom (aug_transforms)
```

### 5. Production Bug & Fix
Encountered a compatibility issue between fast.ai's `aug_transforms` and newer Gradio versions — the pipeline was passing `image + dict` causing a TypeError crash.

**Root cause:** `aug_transforms` saves metadata dict inside the pkl file. Gradio's newer image pipeline conflicted with this.

**Fix:** Bypassed the broken `learn.predict()` wrapper and called `learn.model()` directly with manual preprocessing — same trained weights, clean pipeline.

```python
learn.model.eval()
with torch.no_grad():
    output = learn.model(tensor)
    probs = torch.softmax(output[0], dim=0)
```

---

## 🛠️ Tech Stack

- **Model:** fast.ai 2.8.7 / PyTorch
- **Architecture:** ResNet50 (Transfer Learning)
- **Training:** Kaggle (GPU T4 x2)
- **Deployment:** Gradio + Hugging Face Spaces
- **Dataset:** HAM10000 (10,015 images)

---

## 📁 Project Structure

```
skin-disease-classifier/
│
├── skin_classifier_notebook.ipynb   ← Full training code
├── app.py                           ← Gradio web app
├── requirements.txt                 ← Dependencies
└── README.md                        ← This file
```

---

## 🚀 Run Locally

```bash
pip install fastai gradio
python app.py
```

---

## 📈 Training Progress

| Epoch | Train Loss | Valid Loss | Accuracy | RocAuc |
|---|---|---|---|---|
| freeze 0 | 2.302 | 1.640 | 40.4% | 0.826 |
| freeze 1 | 2.013 | 1.463 | 49.1% | 0.849 |
| freeze 2 | 1.779 | 1.268 | 56.2% | 0.875 |
| 0 | 1.317 | 1.000 | 62.1% | 0.916 |
| 1 | 1.105 | 0.895 | 61.6% | 0.928 |
| 2 | 1.073 | 0.918 | 69.2% | 0.934 |
| 3 | 0.842 | 0.815 | 74.9% | 0.951 |
| 4 | 0.717 | 0.770 | 75.5% | 0.955 |
| 5 | 0.612 | 0.762 | 77.1% | 0.958 |
| 6 | 0.518 | 0.760 | 76.3% | 0.959 |
| **7** | **0.510** | **0.662** | **77.4%** | **0.963** |

---

## ⚠️ Disclaimer

This is a **learning project only**. It is not a certified medical diagnostic tool. Always consult a qualified dermatologist for medical advice. Do not make medical decisions based on this model's output.

---

## 👨‍💻 Author

**Sudeep** — CSE Graduate 2024, PES Institute of Technology and Management  
Building AI/ML projects focused on real-world impact for rural India.

- GitHub: [github.com/Sudeepcpc](https://github.com/Sudeepcpc)
- Live Demo: [huggingface.co/spaces/sudeep1947/skin-disease-classifier](https://huggingface.co/spaces/sudeep1947/skin-disease-classifier)

---

*Built with ❤️ using fast.ai, PyTorch, and Hugging Face*

