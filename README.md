# 🧬 Transcription Factor Binding Site (TFBS) Prediction Web App

Welcome to the **TFBS Prediction Web App**, an advanced and interactive platform designed to predict **Transcription Factor Binding Sites** in DNA sequences. This project integrates multiple machine learning and deep learning models and presents their predictive power through an intuitive web interface built with Streamlit.

---

## 📖 About the Project

Transcription factors are proteins that bind to specific DNA sequences, regulating the expression of genes. These binding sites, known as **Transcription Factor Binding Sites (TFBS)**, play a critical role in cellular processes such as gene regulation, development, and disease mechanisms.

Traditional approaches like motif scanning (e.g., using PWMs) often suffer from low precision and context insensitivity. In this project, we propose a modern, machine-learning-powered alternative using various types of convolutional neural networks (CNNs) and gradient-boosted trees (XGBoost). These models learn patterns from labeled DNA sequences and predict whether a given sequence is likely to be a TFBS or not.

---

## 🎯 Goals

- Accurately classify sequences as TFBS or non-TFBS.
- Compare and showcase different ML/DL approaches:
  - One-hot encoded CNN
  - K-mer embedded CNN
  - Word2Vec-embedded CNN
  - XGBoost with TF-IDF/Count Vectorizer
- Build a unified, easy-to-use web application for real-time predictions.

---

## 🧠 Models Implemented

### 1. CNN with One-Hot Encoding
- Input: 101 bp DNA sequence (strictly required length)
- Encoding: Each nucleotide represented as a 4-dimensional one-hot vector.
- Tuned using Optuna for best performance.
- File: `optuna_cnn_one_hot_utils.py` | Notebook: `Optuna_CNN_One_Hot.ipynb`

### 2. CNN with K-mer Encoding
- Sequences are broken into overlapping 5-mers.
- Vocabulary is generated and tokenized into integers.
- Model is dynamically built and tuned using Optuna.
- File: `optuna_cnn_kmer_utils.py` | Notebook: `Optuna_CNN_k_mer.ipynb`

### 3. CNN with Word2Vec Embedding
- K-mers (k=6) are trained using a Gensim Word2Vec model.
- Embeddings are used in a CNN for classification.
- File: `kmer_2_vec.py` | Notebook: `kmer_2_vec.ipynb`

### 4. XGBoost Classifier
- K-mers converted to feature vectors using Count or TF-IDF vectorizers.
- Trained using GPU-accelerated XGBoost.
- File: `xgb_kmer_utils.py` | Notebook: `XG_Boost.ipynb`

---

## 🌐 Web Application

### Run the app locally:
```bash
streamlit run main_app.py
```

### Features:
- Choose from any of the 3 CNN models.
- Input a raw DNA sequence.
- Real-time prediction with confidence score.
- Warnings for malformed sequences or incorrect input lengths.
- Results rendered with success/error flags for clarity.

---

## 🧪 Training Workflows

Each notebook is self-contained and demonstrates:
- Data preprocessing and loading
- K-mer tokenization / one-hot encoding
- Dataset splitting and padding
- Training with validation and test sets
- Evaluation with metrics:
  - Accuracy
  - ROC-AUC
  - PR-AUC
- Model saving for future inference

---

## 📊 Logging & Experiment Tracking

Use `initialize_results_df.py` to:
- Generate Excel templates for each dataset folder.
- Automatically populate training/testing paths.
- Record accuracy, PR-AUC, and ROC-AUC for comparisons.

---

## 📁 Dataset Format

Each `.data` file should be formatted as:
```
<identifier> <DNA_sequence> <label>
```
- Sequences containing ambiguous nucleotides (e.g., "N") are filtered out automatically.
- Labels must be binary (0 for Non-TFBS, 1 for TFBS).

---

## 🧰 File Overview

| File / Module                  | Description |
|-------------------------------|-------------|
| `main_app.py`                 | Streamlit UI & model interface |
| `load_sequence_data.py`      | Parses DNA sequence files |
| `one_hot.py`                 | One-hot encodes DNA sequences |
| `optuna_cnn_one_hot_utils.py`| One-hot CNN model + training pipeline |
| `optuna_cnn_kmer_utils.py`   | K-mer CNN + Optuna tuning logic |
| `kmer_2_vec.py`              | Word2Vec model and CNN inference |
| `xgb_kmer_utils.py`          | XGBoost feature engineering and training |
| `k_mer_data_loader.py`       | K-mer dataloader |
| `data_loader_one_hot.py`     | One-hot dataloader |
| `initialize_results_df.py`   | Sets up Excel for tracking model results |
| `*.ipynb`                    | Training notebooks for all model types |

---

## 🧑‍🔬 Authors & Credits

**Project By:**  
- 👨‍💻 Harsh Gujarathi  
- 👨‍💻 Dhairya Harshadbhai Chauhan

**Mentors:**  
- 👨‍🏫 Dr. Iman Dehzangi (Rutgers University)  
- 👨‍🏫 Dr. Sunil Shende (Graduate Director)

📧 **Contact:**  
- harshgujarathi07@gmail.com  
- dhairyachauhan619@gmail.com

---

## 🧪 Future Work

- Include attention-based models like Transformers.
- Integration of epigenetic data (e.g., histone marks, DNase-seq).
- AutoML pipeline for automatic model selection.
- Deploy as a web service or bioinformatics plugin.

---

## 📄 License

This project is developed as part of academic coursework and research at **Rutgers University**. For commercial use or publication, please contact the authors for permission.

---

## 📦 Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Launch the app:
```bash
streamlit run main_app.py
```

Enjoy exploring the future of gene regulation with AI! 🧬
