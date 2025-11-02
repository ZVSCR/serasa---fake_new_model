from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import os

app = FastAPI()

# Middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    prediction: str
    confidence: float
    message: str

# ===========================
# 🔹 Carrega BERT e modelo SVM
# ===========================
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Usando dispositivo: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)

# Carrega SVM treinado
model_path = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
svm_model = load(model_path)
print("Modelo SVM carregado com sucesso.")

# ===========================
# 🔹 Função para gerar embedding
# ===========================
def get_bert_embedding(text: str):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).cpu().numpy().flatten()

# ===========================
# 🔹 Rota principal
# ===========================
@app.get("/")
async def root():
    return {"message": "API de Detecção de Fake News com BERTimbau + SVM"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input: TextInput, response: Response):
    try:
        # Evita cache (importante pra Vercel)
        headers = {
            "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
            "CDN-Cache-Control": "no-store",
            "Vercel-CDN-Cache-Control": "no-store"
        }

        text = input.text.strip()
        if not text or len(text) < 15:
            return {
                "prediction": "Indefinido",
                "confidence": 0.0,
                "message": "Texto muito curto para análise confiável. Por favor, insira uma notícia completa."
            }

        # Gera embedding com BERTimbau
        embedding = get_bert_embedding(text)

        # Predição com SVM
        score = svm_model.decision_function([embedding])[0]
        confidence = float(1 / (1 + np.exp(-abs(score))))
        pred = int(score > 0)

        # Mensagens humanizadas
        if pred == 1:
            if confidence > 0.9:
                message = "MODELO MUITO CONFIANTE (Real): Alta certeza de que é uma notícia real."
            elif confidence > 0.75:
                message = "MODELO CONFIANTE (Real): Provável notícia real, mas sempre bom conferir."
            else:
                message = "MODELO INSEGURO (Real): Pende para real, mas com baixa confiança. Verificação recomendada."
        else:
            if confidence > 0.9:
                message = "MODELO MUITO CONFIANTE (Falsa): Forte indicação de conteúdo falso."
            elif confidence > 0.75:
                message = "MODELO CONFIANTE (Falsa): Provável conteúdo enganoso."
            else:
                message = "MODELO INSEGURO (Falsa): Pende para falsa, mas requer verificação humana."

        label = "Fake News" if pred == 0 else "Notícia Real"

        return {
            "prediction": label,
            "confidence": round(confidence, 2),
            "message": message
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar: {str(e)}")

# ===========================
# 🔹 Execução local
# ===========================
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
