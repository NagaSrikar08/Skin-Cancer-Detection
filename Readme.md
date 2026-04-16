# 🔬 Skin Cancer Detection

> An AI-powered deep learning system for early detection and classification of skin cancer lesions from dermoscopic images.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Disclaimer](#disclaimer)
- [License](#license)

---

## 🧠 Overview

Skin cancer is one of the most common and preventable forms of cancer. This project leverages **Convolutional Neural Networks (CNNs)** with **transfer learning** to classify dermoscopic skin lesion images into multiple diagnostic categories, assisting medical professionals in early and accurate detection.

The model is trained on publicly available dermoscopy datasets and achieves competitive performance across multiple lesion types.

---

## ✨ Features

- 🖼️ **Multi-class classification** of skin lesions (up to 7 categories)
- 🔄 **Transfer learning** with pre-trained CNN backbones (ResNet, EfficientNet, DenseNet)
- 📊 **Comprehensive evaluation** — Accuracy, AUC-ROC, F1-Score, Confusion Matrix
- ⚖️ **Class imbalance handling** via weighted loss and oversampling
- 🔍 **Single image inference** for quick predictions
- 📈 **Training visualization** with loss and accuracy curves

---

## 📂 Dataset

The model supports the following publicly available datasets:

| Dataset | Images | Classes | Link |
|---------|--------|---------|------|
| **HAM10000** | 10,015 | 7 | [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T) |
| **ISIC Archive** | 25,000+ | 8 | [isic-archive.com](https://www.isic-archive.com) |

### Lesion Categories

| Label | Code | Description |
|-------|------|-------------|
| 0 | `nv` | Melanocytic nevi |
| 1 | `mel` | Melanoma |
| 2 | `bkl` | Benign keratosis-like lesions |
| 3 | `bcc` | Basal cell carcinoma |
| 4 | `akiec` | Actinic keratoses |
| 5 | `vasc` | Vascular lesions |
| 6 | `df` | Dermatofibroma |

---

## 🗂️ Project Structure

```
Skin_Cancer_Detection/
│
├── data/
│   ├── raw/                    # Original images
│   ├── processed/              # Preprocessed & augmented images
│   └── splits/                 # Train / Val / Test CSV splits
│
├── models/
│   ├── best_model.pth          # Best checkpoint
│   └── final_model.pth         # Final trained model
│
├── notebooks/
│   ├── EDA.ipynb               # Exploratory data analysis
│   └── model_training.ipynb    # Training walkthrough
│
├── src/
│   ├── data_loader.py          # Dataset loading & preprocessing
│   ├── model.py                # Model architecture
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation & metrics
│   └── predict.py              # Inference script
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── classification_report.txt
│
├── config.yaml                 # Hyperparameters & paths
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.8+
- pip or conda

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/your-username/Skin_Cancer_Detection.git
cd Skin_Cancer_Detection

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Key Dependencies

```
torch>=1.12.0
torchvision>=0.13.0
scikit-learn>=1.0
numpy>=1.21
pandas>=1.3
matplotlib>=3.4
Pillow>=9.0
albumentations>=1.2
opencv-python>=4.5
```

---

## 🚀 Usage

### 1. Preprocess Data

```bash
python src/data_loader.py --data_dir data/raw --output_dir data/processed
```

### 2. Train the Model

```bash
python src/train.py --config config.yaml
```

### 3. Evaluate the Model

```bash
python src/evaluate.py --model models/best_model.pth --data_dir data/splits/test
```

### 4. Predict on a Single Image

```bash
python src/predict.py --image path/to/image.jpg --model models/best_model.pth
```

Example output:
```
Prediction: Melanoma (mel)
Confidence: 87.4%
```

---

## 🏗️ Model Architecture

The system uses **transfer learning** with ImageNet pre-trained weights, fine-tuned on skin lesion data.

### Supported Backbones

| Model | Parameters | Top-1 Accuracy (ImageNet) |
|-------|------------|--------------------------|
| ResNet-50 | 25.6M | 76.1% |
| EfficientNet-B3 | 12.2M | 81.6% |
| DenseNet-121 | 8.0M | 74.9% |
| VGG-16 | 138M | 71.6% |

### Training Strategy

- ✅ Pre-trained backbone (frozen initially, then fine-tuned)
- ✅ Custom classification head
- ✅ Data augmentation (flip, rotate, color jitter, zoom, cutout)
- ✅ Dropout regularization
- ✅ Learning rate scheduling (CosineAnnealing / ReduceLROnPlateau)
- ✅ Class-weighted cross-entropy loss

---

## 📊 Results

> *(Update this section with your actual results after training)*

| Model | Accuracy | AUC-ROC | F1 (macro) |
|-------|----------|---------|------------|
| ResNet-50 | --% | -- | -- |
| EfficientNet-B3 | --% | -- | -- |
| DenseNet-121 | --% | -- | -- |

---

## ⚠️ Disclaimer

> **This project is intended for RESEARCH and EDUCATIONAL purposes only.**
>
> It is **NOT** a certified medical device and should **NOT** be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified and licensed healthcare provider for any medical concerns.

---

## 👥 Contributors

- **Author:** *(Your Name)*
- **Version:** 1.0.0
- **Date:** 2026

Contributions, issues, and feature requests are welcome! Feel free to open a pull request or issue.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

<p align="center">Made with ❤️ for early cancer detection</p>
