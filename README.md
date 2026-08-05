# NLP Question Answering System

## Overview

This project is an NLP-based Question Answering System built using Transformer models. The system takes a context paragraph and a user question, then extracts the most relevant answer from the provided text.

## Features

* Transformer-based Question Answering
* Fine-tuned DistilBERT model
* Confidence score for predictions
* Interactive Streamlit web interface
* Evaluation using Exact Match (EM) and F1 Score

## Technologies Used

* Python
* Hugging Face Transformers
* PyTorch
* Streamlit
* Scikit-learn

## Project Structure

```text
NLP_Question_Answering_System/
│
├── models/
│   └── qa_model_deploy/
│
├── predict.py
├── evaluate.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

## Example

Context:
Python is a high-level programming language created by Guido van Rossum.

Question:
Who created Python?

Answer:
Guido van Rossum

## Evaluation

Sample Evaluation Results:

* Exact Match (EM): 100%
* F1 Score: 100%

Note: These results were obtained on a small demonstration evaluation set.

## Future Improvements

* Compare DistilBERT with BERT and RoBERTa
* Deploy the application online
* Improve evaluation using a larger test set
* Support multiple languages