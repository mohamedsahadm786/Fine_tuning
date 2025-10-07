# Confidence Classification using Fine-Tuned VideoMAE and WavLM

## 📌 Project Overview
This project focuses on **predicting speaker confidence levels** (Low, Moderate, High) from short interview video clips.  
The models were developed as part of an **AI-powered video interview trainer**, where user responses are analyzed to provide intelligent feedback on both audio and video components.

---

## 🎯 Motivation
**Fine-tuning pre-trained models** has become a key technique in modern AI.  
Instead of training large models from scratch, adapting existing models to domain-specific tasks allows for efficient and high-quality solutions, especially when data availability is limited.  

In this project, fine-tuning was applied to **VideoMAE** (for video) and **WavLM** (for audio) to create robust **multimodal confidence classification models**.

---

## 🧪 Dataset Details
- **Original dataset size:** 42 videos  
- **Problem:** The dataset was **small and imbalanced** across the three confidence classes.  
- **Augmentation:** Applied multiple augmentation techniques to expand the dataset to **79 videos**.  
- **Effect:** After augmentation, class imbalance was effectively minimized, improving training stability and validation performance.

---

## 🏗️ Model Development
A total of **8–9 different models** were developed by experimenting with various techniques:

### 1. Pooling Strategies
- **Mean Pooling**  
- **Mean + Standard Deviation Pooling**  
- **Attention Pooling**

These methods were used to convert multiple clip-level embeddings into a single video-level representation.

### 2. Cross-Validation Methods
- **5-Fold Cross-Validation**  
- **Group K-Fold Cross-Validation**

Different validation strategies were explored to handle the small dataset effectively and reduce overfitting risks.

### 3. Embedding Fusion
For each video:
- Video features were extracted using **VideoMAE**.  
- Audio features were extracted using **WavLM**.  
- These embeddings were pooled, fused, and passed through an **MLP classifier** for **ordinal confidence prediction** (0 = Low, 1 = Moderate, 2 = High).

---

## ⚡ Key Challenges & Solutions

| **Challenge**                  | **Approach**                                                                 |
|----------------------------------|-------------------------------------------------------------------------------|
| Small dataset                    | Data augmentation to expand dataset size                                     |
| Class imbalance                  | Balanced augmentation strategies to equalize class distribution              |
| Overfitting risk                 | Robust cross-validation, pooling strategies, and careful model design        |
| Multimodal fusion complexity     | Separate audio/video embeddings followed by fusion and joint fine-tuning     |

---

## 📈 Results
Despite working with a small dataset, the models achieved **strong validation performance without overfitting**.  
The best results were obtained using a combination of **attention pooling** and **group K-fold validation**.

---

## 🗂️ Repository Contents
- `models/` – Saved weights of each trained model  
- `notebooks/` – Experiment notebooks for each pooling and validation technique  
- `csv/` – To get an idea about the data set structure
- `kaggle notebook/` – To get an idea about how I augmented from 42 to 79 by removing the class imbalance
- `pdf/` - The aechitecture of the pipeline
- `README.md` – Project overview

---

## 🔗 Further Details
For detailed implementation, model weights, and experiment logs, please explore the repository files.

---

## 📝 Citation
If you find this project useful, consider citing or referencing it in your work.

---

## 📬 Contact
For any queries, feel free to reach out via LinkedIn or GitHub Issues.
