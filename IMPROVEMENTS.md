# 🔧 Improvements and Iterative Development

This document outlines the progression of the CRNN-based artwork classification model from the initial baseline to the current improved version.

---

## 📊 1. Performance Improvements

![Performance Graph](results/images/performance_graph.png)

### Baseline:
- EfficientNet-B0 + RNN
- Limited tuning
- Lower Macro-F1 and unstable validation

### Improved Model:
- ConvNeXt / DINOv2 backbone
- Multi-task learning (Artist + Style + Genre)
- Better convergence and stability

👉 Major improvement observed in **Style classification**, which was the most challenging task.

---

## 🧠 2. Architectural Improvements

![Architecture Comparison](results/images/architecture_comparison.png)

### Changes:
- Replaced EfficientNet with ConvNeXt / DINOv2
- Introduced multi-task output heads
- Added Bi-LSTM for sequence modeling

### Impact:
- Stronger feature extraction  
- Shared learning across tasks  
- Improved generalization  

---

## 🔄 3. Workflow Enhancements

![Workflow](results/images/workflow_final.png)

- Structured training pipeline
- Better validation tracking (Macro-F1 + Accuracy)
- Iterative experimentation for optimization

---

## 🎯 4. Improved Feature Representation

![System Overview](results/images/artwork_pipeline_v3.png)

- Shared representation learning across tasks
- Better encoding of visual and contextual features
- Improved prediction consistency

---

## 🔍 5. Improved Classification Behavior

![Confusion Matrix](results/images/confusion_matrix.png)

- Stronger diagonal → better accuracy  
- Reduced misclassification across classes  
- Improved separation of similar styles  

---

## 🧪 6. Methods Explored

During development, multiple approaches were tested:

### 1. Baseline CRNN (EfficientNet + RNN)
- Simple pipeline
- Limited performance

### 2. ConvNeXt Backbone
- Improved spatial feature extraction

### 3. DINOv2 (Self-Supervised Learning)
- Strong visual representation
- Better generalization

### 4. Multi-task Learning
- Joint prediction (Artist, Style, Genre)
- Shared feature learning

### 5. Training Optimizations
- Label smoothing  
- Cosine LR scheduling  
- Improved regularization  

---

## 📌 7. Key Learnings

- Style classification is inherently harder than artist classification  
- Strong backbones significantly improve performance  
- Multi-task learning enhances generalization  
- Data quality (especially labels) is critical  

---

## 🔮 8. Future Improvements

- Ensemble models (CLIP + DINOv2)  
- Test-time augmentation (TTA)  
- Contrastive learning  
- Larger-scale training  

---

## 🧾 9. Summary

This project reflects an **iterative improvement process**, focusing on:

- Identifying weaknesses  
- Testing multiple architectures  
- Optimizing training strategies  
- Selecting the best-performing approach  

The final model demonstrates improved performance, stability, and generalization across all tasks.