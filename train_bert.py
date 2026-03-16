# src/train_bert.py
import torch
from torch import nn
from torch.optim import AdamW
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from bert_prep import train_loader, test_loader
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

NUM_CLASSES = 16
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=NUM_CLASSES
)

# ✅ LOAD PREVIOUSLY TRAINED MODEL
model.load_state_dict(
    torch.load("model/bert_mbti_model.pt", map_location=device)
)

model = model.to(device)


# Optimizer, scheduler, loss
LEARNING_RATE = 2e-5
EPOCHS = 1  # 1 epoch at a time for CPU
optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=0, num_training_steps=total_steps
)
loss_fn = nn.CrossEntropyLoss().to(device)

# Training function
def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler):
    model.train()
    total_loss = 0
    for i, batch in enumerate(data_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        if (i+1) % 10 == 0:
            print(f"Batch {i+1}/{len(data_loader)} | Current Loss: {loss.item():.4f}")
            
    return total_loss / len(data_loader)

# Evaluation function
def eval_model(model, data_loader, device):
    model.eval()
    correct_predictions = 0
    total = 0
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            correct_predictions += (preds == labels).sum().item()
            total += labels.size(0)
    return correct_predictions / total

# Training loop
train_loss = train_epoch(model, train_loader, loss_fn, optimizer, device, scheduler)
val_acc = eval_model(model, test_loader, device)
print(f"Epoch 1/1 | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")

# Save model
os.makedirs("model", exist_ok=True)
torch.save(model.state_dict(), "model/bert_mbti_model.pt")
print("Model trained and saved at 'model/bert_mbti_model.pt'!")
