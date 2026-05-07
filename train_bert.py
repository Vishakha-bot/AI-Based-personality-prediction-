import torch
from torch import nn
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from bert_prep import train_loader, test_loader
import os

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Model
NUM_CLASSES = 16
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=NUM_CLASSES
)

# Load saved model if exists
model_path = "model/bert_mbti_model.pt"

if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    print("Loaded existing model ✅")
else:
    print("Training from scratch 🚀")

model = model.to(device)

# Hyperparameters
LEARNING_RATE = 2e-5
EPOCHS = 3   # 🔥 now multiple epochs in one run

optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

loss_fn = nn.CrossEntropyLoss().to(device)

# Training function
def train_epoch(model, data_loader):
    model.train()
    total_loss = 0

    for i, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        loss = outputs.loss
        total_loss += loss.item()

        loss.backward()
        optimizer.step()
        scheduler.step()

        if (i + 1) % 10 == 0:
            print(f"Batch {i+1}/{len(data_loader)} | Loss: {loss.item():.4f}")

    return total_loss / len(data_loader)

# Evaluation
def eval_model(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return correct / total

# 🚀 TRAINING LOOP
print("\nStarting Training...\n")

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")

    train_loss = train_epoch(model, train_loader)
    val_acc = eval_model(model, test_loader)

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

# Save model
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), model_path)

<<<<<<< HEAD
print("\nModel saved successfully ✅")

   
=======
print("\nModel saved successfully ✅")
>>>>>>> 1185519 (Updated preprocessing, training, prediction and requirements)
