# NLP Question Answering System

## Overview

This project is an NLP-based Question Answering System built using Transformer models and Hugging Face Transformers. The system takes a context paragraph and a user question, then extracts the most relevant answer directly from the provided text.

The project demonstrates the complete NLP pipeline, including data preprocessing, model fine-tuning, evaluation, prediction, and deployment through an interactive Streamlit web application.

---

## Features

* Transformer-based Question Answering
* Fine-tuned DistilBERT model
* Context-aware answer extraction
* Confidence score for predictions
* Interactive Streamlit interface
* Evaluation using Exact Match (EM) and F1 Score
* End-to-end NLP workflow

---

## Technologies Used

* Python
* Hugging Face Transformers
* PyTorch
* Streamlit
* Pandas
* NumPy
* Scikit-learn

---

## Project Structure

```text
NLP_Question_Answering_System/
│
├── data/
│   ├── train-v1.1.json
│   └── dev-v1.1.json
│
├── results/
│   └── squad_processed.csv
│
├── preprocess.py
├── train.py
├── predict.py
├── evaluate.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## Project Workflow

### 1. Data Preprocessing

The dataset is cleaned and transformed into a format suitable for training Transformer-based Question Answering models.

### 2. Model Training

A DistilBERT model is fine-tuned on a Question Answering dataset to learn how to locate answers within a given context.

### 3. Evaluation

The model performance is evaluated using:

* Exact Match (EM)
* F1 Score

### 4. Prediction

Users can provide a context and ask questions. The model extracts the answer directly from the supplied text.

### 5. Deployment

The system is deployed through a Streamlit web interface for interactive usage.

---

## Web Application

The project includes an interactive Streamlit application where users can:

* Enter a context paragraph
* Ask questions about the text
* Receive extracted answers
* View confidence scores

---

## Example

### Context

Python is a high-level programming language created by Guido van Rossum and first released in 1991.

### Question

Who created Python?

### Predicted Answer

Guido van Rossum

---

## Sample Evaluation Results

* Exact Match (EM): 100%
* F1 Score: 100%

**Note:** These results were obtained on a small demonstration evaluation set and are intended to validate the evaluation pipeline.

---

## Model

Fine-tuned DistilBERT model for Extractive Question Answering.

---

## Installation

```bash
pip install -r requirements.txt
```

---

## Run Streamlit Application

```bash
streamlit run streamlit_app.py
```

---

## Future Improvements

* Compare DistilBERT with BERT and RoBERTa
* Deploy the application online
* Improve evaluation using larger benchmark datasets
* Add multilingual question answering support
* Experiment with advanced Transformer architectures

---

## Note

The trained model files are not included in this repository due to GitHub file size limitations. The repository contains the complete source code, preprocessing pipeline, training scripts, evaluation scripts, and deployment interface.

---

AI Engineering Student | Machine Learning | Deep Learning | NLP | Computer Vision
