# Plant Leaf Disease Detection using ResNet50, PCA, and SVM 🍃
Reproducing a hybrid deep learning model for plant leaf disease detection on the PlantVillage dataset.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)

## 📌 Project Overview
This repository contains a complete, single-notebook reproduction of the research paper: **"A hybrid deep learning model for robust and efficient plant leaf disease detection using ResNet50, PCA, and SVM"**. 

The project aims to classify plant leaf diseases across **38 different classes** using the publicly available **PlantVillage dataset** (approx. 54k images).

## 🚀 Methodology Pipeline
Our end-to-end machine learning pipeline consists of:
1. **Data Preprocessing:** Resizing (224x224) and Normalizing images.
2. **Feature Extraction:** Using a pre-trained **ResNet50** (with classification layer removed) to extract high-level semantic features (2048 dimensions).
3. **Dimensionality Reduction:** Applying **Principal Component Analysis (PCA)** to reduce the feature vectors to 256 components, reducing computational cost and preventing overfitting.
4. **Classification:** Training a **Linear Support Vector Machine (SVM)** on the reduced features.

## 📊 Results & Performance
The reproduced model demonstrated exceptional stability and outperformed the baseline mentioned in the original paper, achieving highly robust metrics during evaluation:

- **5-Fold Cross-Validation Accuracy:** `97.36% (± 0.0029)`
- **Final Test Accuracy:** `97.35%`
- **Precision:** `97.37%`
- **Recall:** `97.35%`
- **F1-Score:** `97.35%`

### Confusion Matrix

<p align="center">
  <img src="Confusion_Matrix.png" alt="Confusion Matrix" width="900">
</p>

## 📈 Performance Comparison: Original Paper vs. Our Implementation

The table below presents a direct comparison between the evaluation metrics reported in the original paper (the baseline model without red-mask segmentation) and the final results achieved by our single-notebook pipeline:

| Metric | Original Paper Baseline | Our Implementation (Test Set) | Our Implementation (5-Fold CV Mean) |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 96.08% | **97.35%** | **97.36%** |
| **Precision** | 96.08% | **97.37%** | — |
| **Recall** | 96.08% | **97.35%** | — |
| **F1-Score** | 96.06% | **97.35%** | — |

### 🔍 Analysis of Performance Improvement
Our implementation achieved a noticeable boost in overall performance compared to the original paper's findings. This improvement can be attributed to the following technical factors:

1. **Upgraded Pretrained Feature Extractor (ResNet50 V2):** The original paper was built using older iterations of the ResNet50 pretrained weights. In this pipeline, we utilized `ResNet50_Weights.DEFAULT`, which incorporates PyTorch's modern **V2 training recipe** (achieving higher Top-1 accuracy on ImageNet). This allowed our backbone to extract significantly richer and more distinct semantic features from the leaf images.
2. **Optimal Dimensionality Reduction:** Compressing the 2048-dimensional deep feature maps down to exactly **256 Principal Components** via PCA effectively discarded background noise and redundant spatial information while retaining maximum variance.
3. **Robust Generalization via Linear SVM:** The high similarity between our 5-Fold Cross-Validation accuracy (`97.36%`) and the final Test Set accuracy (`97.35%`), alongside a tiny standard deviation (`±0.0029`), mathematically proves that the model successfully learned generalizable disease patterns rather than overfitting to the dataset.

## 💻 How to Run
Everything is contained within a single, easy-to-follow Jupyter Notebook. 

1. Open `r_hybrid__.ipynb` in Google Colab.
2. Upload your `kaggle.json` to directly download the dataset.
3. Run the cells sequentially. The notebook includes automated saving of extracted features to Google Drive to prevent data loss upon runtime disconnection.

## 📄 References

- Original Paper: **[A hybrid deep learning model for robust and efficient plant leaf disease detection using ResNet50, PCA, and SVM](https://www.nature.com/articles/s41598-026-46085-w)**
- Authors: Saba Begum, Naresh E, Srinidhi N. N.
- Dataset: [PlantVillage Dataset on Kaggle](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
