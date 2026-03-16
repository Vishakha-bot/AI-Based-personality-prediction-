import torch
from transformers import BertTokenizer, BertForSequenceClassification
from bert_prep import classes

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "model/bert_mbti_model.pt"
MAX_LEN = 64

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased",
    num_labels=len(classes)
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# 🔥 PRIORITY-BASED RULES (ORDER MATTERS)
RULES = [
    # Leadership / Social dominance
    (["lead", "leader", "leading", "command"], "ENTJ"),
    (["motivate", "motivating", "guide", "inspire"], "ENFJ"),
    (["efficiency", "results", "deadline"], "ESTJ"),

    # Social / Fun
    (["fun", "enjoying", "living in the moment", "experience"], "ESFP"),
    (["people", "group", "community"], "ENFJ"),

    # Thinking / Logic
    (["logic", "analyze", "reason"], "INTJ"),
    (["rules", "structured", "organized"], "ISTJ"),

    # Creativity
    (["ideas", "imagine", "creative"], "ENFP"),

    # Emotional (LOW priority)
    (["feel", "emotion", "emotional", "attachment"], "INFP"),
]

def rule_based(text):
    text = text.lower()
    for keywords, mbti in RULES:
        for key in keywords:
            if key in text:
                return mbti
    return None


def predict_personality(text):
    # 1️⃣ Rule-based FIRST
    rule_pred = rule_based(text)
    if rule_pred:
        return rule_pred, "rule-based"

    # 2️⃣ BERT fallback
    encoding = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LEN,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(
            input_ids=encoding["input_ids"].to(device),
            attention_mask=encoding["attention_mask"].to(device)
        )
        pred = torch.argmax(outputs.logits, dim=1).item()

    return classes[pred], "bert"

# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":
    print("\n🧠 Hybrid Personality Predictor (FIXED)\n")

    while True:
        text = input("Enter text (or exit): ")
        if text.lower() == "exit":
            break

        label, source = predict_personality(text)
        print(f"✨ Prediction: {label}  ({source})\n")
