<p align="center">
  <img src="assets/fmd-banner.svg" alt=<p align="center">
  <img src="assets/fmd-banner.svg" alt="Face Mask Detection CNN Banner" width="100%" />
</p>

<h1 align="center">Face Mask Detection (CNN)</h1>
 width="100%" />
</p>

<h1 align="center">Face Mask Detection (CNN)</h1>

A clean, production-style PyTorch implementation for **binary face mask classification**:


- `mask` vs `no_mask`
- Lightweight **CNN** (no internet needed for pretrained weights)
- Reproducible training, early stopping, evaluation, and CLI inference

## 📦 Folder Layout

```
face-mask-detection-cnn/
├── src/
│   ├── config.yaml    # hyperparameters & paths
│   ├── dataset.py     # loaders & transforms
│   ├── model.py       # CNN model
│   ├── train.py       # training loop + validation & test
│   ├── infer.py       # single-image prediction
│   └── utils.py       # metrics, plots, seeding
├── models/            # saved weights will appear here
├── data/              # put dataset here (see below)
├── requirements.txt
└── README.md
```

## 🗄️ Dataset (Kaggle Recommended)

Use the Kaggle dataset: **Face Mask 12K Images Dataset**  
Author: *Ashish Jangra*

After download, arrange folders like this (already compatible with this repo):

```
data/
├── train/
│   ├── mask/
│   └── no_mask/
├── val/
│   ├── mask/
│   └── no_mask/
└── test/
    ├── mask/
    └── no_mask/
```

> If your download has different names (e.g., `with_mask`, `without_mask`), just rename to `mask` and `no_mask` or update `classes` in `src/config.yaml`.

## ⚙️ Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 🏋️ Train

```bash
python src/train.py
```
- Hyperparameters live in `src/config.yaml`.
- Best model is saved to: `models/best_mask_cnn.pt`.

## 🔎 Infer (Single Image)

```bash
python src/infer.py --image path/to/image.jpg
```
Output example:
```
Prediction: mask | Probabilities: {'mask': 0.9871, 'no_mask': 0.0129}
```

## 📈 Results (Fill After Training)

| Metric | Value |
|---|---|
| Val Accuracy | __ |
| Test Accuracy | __ |
| Model Params | ~3.2M |

## 👤 Author

**Md. Prince Mahmud**  
M.Sc. Computer Science, Philipps-Universität Marburg  
GitHub: https://github.com/mahmudprince98
