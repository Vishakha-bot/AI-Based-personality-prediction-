# src/bert_prep.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

# -------------------------------
# 1️⃣ Load dataset
# -------------------------------
df = pd.read_csv("data/mbti_1.csv")  # Your CSV file
print("Dataset loaded successfully!")
print(df.head())

# -------------------------------
# 2️⃣ Encode MBTI labels
# -------------------------------
le = LabelEncoder()
df['label'] = le.fit_transform(df['type'])
classes = le.classes_
print("Labels encoded successfully!")
print("Classes:", classes)

# -------------------------------
# 3️⃣ Train/test split
# -------------------------------
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
print(f"Original Training samples: {len(train_df)}, Test samples: {len(test_df)}")

# -------------------------------
# 3a️⃣ Reduce number of samples for CPU speed
# -------------------------------
train_df = train_df.sample(2000, random_state=42)
test_df = test_df.sample(500, random_state=42)
print(f"Sampled Training samples: {len(train_df)}, Test samples: {len(test_df)}")

# -------------------------------
# 4️⃣ Tokenizer & settings
# -------------------------------
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

MAX_LEN = 64   # smaller for CPU
BATCH_SIZE = 4  # smaller batch for CPU

# -------------------------------
# 5️⃣ Dataset class
# -------------------------------
class PersonalityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = int(self.labels[idx])
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# -------------------------------
# 6️⃣ Create datasets & DataLoaders
# -------------------------------
train_dataset = PersonalityDataset(
    texts=train_df['posts'].tolist(),
    labels=train_df['label'].tolist(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

test_dataset = PersonalityDataset(
    texts=test_df['posts'].tolist(),
    labels=test_df['label'].tolist(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print("PyTorch datasets ready!")
print("DataLoaders ready! You can now feed them into a BERT model.")

# -------------------------------
# 7️⃣ Export for train_bert.py
# -------------------------------
__all__ = ['train_loader', 'test_loader', 'train_df', 'test_df', 'classes']
