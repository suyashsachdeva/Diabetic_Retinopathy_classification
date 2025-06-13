
# Diabetic Retinopathy Classification 🩺

A deep learning pipeline for **automated diabetic retinopathy (DR)** detection and severity grading—from preprocessing fundus images to model training and evaluation.

## 🚀 Features
- Multi-class classification (No DR, Mild, Moderate, Severe, Proliferative DR)  
- Uses convolutional neural networks (DenseNet, EfficientNet, etc.)  
- Data augmentation: flips, rotations, brightness/contrast adjustments  
- Custom metrics: accuracy, precision, recall, AUC  
- GPU-accelerated training  
- Pre-trained weights support and fine-tuning options  

## 📂 Table of Contents
1. [Installation](#installation)  
2. [Dataset](#dataset)  
3. [Usage](#usage)  
4. [Examples](#examples)  
5. [Project Structure](#project-structure)  
6. [Configuration](#configuration)  
7. [Troubleshooting](#troubleshooting)  
8. [License](#license)  
9. [Contact](#contact)  
10. [References](#references)

---

## Installation

1. Clone the repo:

   ```bash
   git clone https://github.com/suyashsachdeva/Diabetic_Retinopathy_classification.git
   cd Diabetic_Retinopathy_classification
```

2. (Recommended) Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## Dataset 📊

We recommend using the **EyePACS dataset** from the Kaggle “Diabetic Retinopathy Detection” competition—it includes high-resolution retinal images labeled across five classes ([arxiv.org][1], [paperswithcode.com][2], [kaggle.com][3]):

**Link:**
[https://www.kaggle.com/c/diabetic-retinopathy-detection/data](https://www.kaggle.com/c/diabetic-retinopathy-detection/data)

Alternatively, you can use the **“Diabetic Retinopathy Dataset”** (1 000 healthy, 370 mild, 900 moderate, 190 severe, 290 proliferative) from Kaggle ([kaggle.com][4], [kaggle.com][5]).

### Example download:

```bash
kaggle competitions download -c diabetic-retinopathy-detection
unzip train.zip -d data/retina
```

---

## Usage

### 1. Data Preparation

Split and augment images:

```bash
python scripts/prepare_data.py \
  --input_dir data/retina/train \
  --output_dir data/processed \
  --img_size 224 \
  --val_split 0.2
```

### 2. Train Model

Customize hyperparameters:

```bash
python train.py \
  --data_dir data/processed \
  --model densenet121 \
  --epochs 25 \
  --batch_size 16 \
  --lr 1e-4 \
  --output_dir models/dense_dr
```

### 3. Evaluate

Generate metrics and confusion matrix on test set:

```bash
python evaluate.py \
  --model_path models/dense_dr/best.pth \
  --data_dir data/processed/test
```

### 4. Predict

Classify a single image:

```bash
python predict.py \
  --model_path models/dense_dr/best.pth \
  --input_image samples/test_retina.jpeg
```

---

## 📸 Examples

Prior to/after classification outputs and confusion matrices are available in the `/results/` folder.

---

## Project Structure

```text
Diabetic_Retinopathy_classification/
├── data/
│   └── retina/
├── scripts/
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
├── models/
│   └── dense_dr/
├── requirements.txt
└── README.md
```

---
## 🛠️ Troubleshooting

* **CUDA / GPU errors:** Check GPU driver compatibility and `torch.cuda.is_available()`.
* **Overfitting:** Increase augmentation or add dropout.
* **Class imbalance:** Use weighted loss or oversample minority classes.

---

## ⚖️ License

Licensed under **MIT**. See [LICENSE](LICENSE) for details.

---

## 📬 Contact

For issues or feature requests, open an [issue](https://github.com/suyashsachdeva/Diabetic_Retinopathy_classification/issues), or reach out to **Suyash Sachdeva** via email.

---
