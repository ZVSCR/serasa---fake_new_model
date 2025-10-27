from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# uvicorn main:app --reload

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

app = FastAPI(title="Fake News Classifier API - BERTimbau")

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
    text = input.text

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()

    label = "Fake News" if predicted_class == 0 else "Notícia Real"
    return {"prediction": label}

@app.get("/")
def root():
    return {"message": "API de classificação de Fake News com BERTimbau está online"}
