# 🎨 CRNN Submission – Artwork Classification (ArtExtract)

This project focuses on **multi-task artwork classification** using deep learning, where a single model jointly predicts:

- 🧑‍🎨 Artist
- 🎭 Style
- 🖼️ Genre

The system leverages **strong visual feature extraction** and **shared representation learning** to improve performance across all tasks.

---

## 📌 1. System Overview

![System Overview](results/images/artwork_pipeline_v3.png)

The pipeline follows a structured deep learning workflow:

- Input artwork image
- Preprocessing (resize, normalization, augmentation)
- Feature extraction using CNN backbone (ConvNeXt / DINOv2)
- Sequence modeling using Bi-LSTM
- Multi-task predictions (Artist, Style, Genre)

---

## 🧠 2. Model Architecture

![Model Architecture](results/images/model_architecture.png)

The architecture combines:

- **CNN Backbone** → Extracts visual features  
- **Feature Map + Flattening** → Converts spatial features  
- **Bi-LSTM** → Captures sequential dependencies  
- **Fully Connected Layers** → Final transformations  
- **Multi-task Heads** → Artist, Style, Genre prediction  

---

## 🔄 3. Workflow

![Workflow](results/images/workflow_final.png)

The training and evaluation pipeline includes:

- Dataset preparation  
- Preprocessing and augmentation  
- Model training (multi-task learning)  
- Validation using accuracy and Macro-F1  
- Final testing and result generation  

---

## 📊 4. Performance Analysis

![Performance Graph](results/images/performance_graph.png)

The model shows consistent improvement over baseline approaches:

- Stronger backbones (ConvNeXt / DINOv2)
- Improved convergence behavior
- Better validation stability

---

## 🔍 5. Confusion Matrix (Style Classification)

![Confusion Matrix](results/images/confusion_matrix.png)

- Strong diagonal indicates correct predictions  
- Reduced off-diagonal errors show improved class separation  
- Remaining confusion occurs in visually similar styles  

---

## ⚖️ 6. Architecture Comparison

![Architecture Comparison](results/images/architecture_comparison.png)

Comparison between baseline and improved model:

- Transition from **single-task → multi-task learning**
- Stronger feature extraction (ConvNeXt / DINOv2)
- Improved Macro-F1 and generalization

---

## 🚀 7. Key Highlights

- Multi-task learning improves performance across all tasks  
- Strong CNN backbones enhance feature representation  
- Sequence modeling captures contextual relationships  
- Improved validation metrics (especially Style classification)  

---

## 📄 8. Improvements Over Previous Version

👉 See [`IMPROVEMENTS.md`](IMPROVEMENTS.md) for a detailed explanation of:

- Model evolution  
- Experiments performed  
- Performance improvements  

---

## 🔮 9. Future Work

- Ensemble models (CLIP + DINOv2)  
- Test-time augmentation (TTA)  
- Better class imbalance handling  
- Contrastive learning approaches  

---

## 📎 License

MIT License