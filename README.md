## Dataset

The dataset used in this project is the MBTI Personality Dataset.

Due to large file size, the dataset is not included in this repository.
Dataset source:
https://www.kaggle.com/datasets/datasnaek/mbti-type
## How to Run

1. Download dataset from Kaggle link
2. Place dataset in project folder
3. Run baseline_models.ipynb
4. Run bert_model.py for transformer model
This project predicts MBTI personality types from text using a hybrid approach:
> BERT (Transformer model)
> Rule-based keyword matching

Features
BERT-based classification (16 classes)
Hybrid prediction (rule-based + BERT)
CLI-based prediction (predict.py)
Model saving & resume training

Tech Stack
Python, PyTorch
Transformers (BERT)
Pandas, Scikit-learn

