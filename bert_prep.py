import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

print("Loading dataset...")

# Load dataset
df = pd.read_csv("data/mbti_1.csv")

# Keep only required columns
df = df[['type', 'posts']]
df.dropna(inplace=True)

print("Dataset loaded successfully!")
print(df.head())

# Encode labels (16 MBTI types)
labels = sorted(df['type'].unique())
label_dict = {label: i for i, label in enumerate(labels)}
df['label'] = df['type'].map(label_dict)

print("Labels encoded successfully!")
print("Classes:", labels)

# Train-test split
train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)

print(f"Original Training samples: {len(train_df)}, Test samples: {len(test_df)}")

# ⚡ Reduce size for faster CPU training (optional but recommended)
train_df = train_df.sample(n=2000, random_state=42)
test_df = test_df.sample(n=500, random_state=42)

print(f"Sampled Training samples: {len(train_df)}, Test samples: {len(test_df)}")

# Tokenizer (UPDATED)
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

MAX_LEN = 128
BATCH_SIZE = 8

# Dataset class
class MBTIDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create datasets
train_dataset = MBTIDataset(
    texts=train_df['posts'].tolist(),
    labels=train_df['label'].tolist(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

test_dataset = MBTIDataset(
    texts=test_df['posts'].tolist(),
    labels=test_df['label'].tolist(),
    tokenizer=tokenizer,
    max_len=MAX_LEN
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print("PyTorch datasets ready!")
print("DataLoaders ready! You can now feed them into a BERT model 🚀")



