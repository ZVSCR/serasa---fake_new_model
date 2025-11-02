from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModel
import torch
from joblib import load
import numpy as np
import os
import sys

# Adiciona o diretório atual ao path para imports
sys.path.append(os.path.dirname(__file__))

MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
device = "cuda" if torch.cuda.is_available() else "cpu"

try:
    from utils.preprocess import preprocess_text
except ImportError:
    # Fallback simples para deploy
    def preprocess_text(text):
        return text.lower().strip()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)

async def get_bert_embedding(text: str):
    """Transforma o texto em embedding BERT médio"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)

    # Usa a média das ativações como representação vetorial
    embedding = outputs.last_hidden_state.mean(dim=1)
    return await embedding.cpu().numpy()

app = FastAPI(
    title="API de Detecção de Fake News (SVM com embeddings BERT)",
    version="1.0.0"
)

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

# Cache do modelo
_model = None

@app.get("/")
async def root():
    return {"message": "API de Detecção de Fake News — modelo leve com embeddings BERT + SVM"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input: TextInput, response: Response):
    try:
        # Controle de cache para a Vercel
        headers = {
            "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
            "CDN-Cache-Control": "no-store",
            "Vercel-CDN-Cache-Control": "no-store"
        }

        global _model
        if _model is None:
            model_dir = os.path.join(os.path.dirname(__file__), "model")
            model_path = os.path.join(model_dir, "model.pkl")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
            _model = load(model_path)

        model = _model
        text = input.text.strip()

        if not text or len(text) < 15:
            return {
                "prediction": "Indefinido",
                "confidence": 0.0,
                "message": "O texto é muito curto para uma análise confiável. Por favor, insira uma notícia completa."
            }

        # Pré-processamento leve
        embedding = get_bert_embedding(input.text)

        # Predição com SVM (o modelo já foi treinado sobre embeddings)
        score = model.decision_function(embedding)[0]
        confidence = float(1 / (1 + np.exp(-abs(score))))
        pred = int(score > 0)

        # Mensagens interpretáveis
        if pred == 1:
            if confidence > 0.9:
                message = "Essa notícia parece altamente confiável."
            elif confidence > 0.75:
                message = "Provável notícia real, mas é bom conferir as fontes."
            else:
                message = "O modelo pende para real, mas com baixa confiança."
        else:
            if confidence > 0.9:
                message = "Forte indicação de que esta notícia é falsa."
            elif confidence > 0.75:
                message = "Provável conteúdo falso, mas recomenda-se verificar fontes."
            else:
                message = "O modelo pende para falsa, mas sem alta confiança."

        label = "Fake News" if pred == 0 else "Notícia Real"

        return {
            "prediction": label,
            "confidence": round(confidence, 2),
            "message": message
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar: {str(e)}")

# Execução local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
