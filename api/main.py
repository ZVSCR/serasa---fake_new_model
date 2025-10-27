from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
import numpy as np
import os
import sys

# Adiciona o diretório atual ao path para imports
sys.path.append(os.path.dirname(__file__))

try:
    from utils.preprocess import preprocess_text
except ImportError:
    # Fallback para desenvolvimento
    def preprocess_text(text):
        return text.lower().strip()

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

# Cache do modelo
_model = None

@app.get("/")
async def root():
    return {"message": "API de Detecção de Fake News"}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input: TextInput, response: Response):
    try:
        headers = {
            "Cache-Control": "no-store, no-cache, must-revalidate, proxy-revalidate",
            "CDN-Cache-Control": "no-store",
            "Vercel-CDN-Cache-Control": "no-store"
        }

        global _model
        if _model is None:
            model_dir = os.path.join(os.path.dirname(__file__), "model")
            model_path = os.path.join(model_dir, "model.pkl")
            _model = load(model_path)

        model = _model
        cleaned_text = preprocess_text(input.text)

        # Probabilidade ou score
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba([cleaned_text])[0]
            confidence = float(np.max(proba))
            pred = int(np.argmax(proba))
        else:
            # Fallback para modelos sem predict_proba (como LinearSVC)
            score = model.decision_function([cleaned_text])[0]
            confidence = float(1 / (1 + np.exp(-abs(score)))) 
            pred = int(score > 0)

        # Mensagens mais humanas
        if pred == 1:
            if confidence > 0.90:
                message = "MODELO MUITO CONFIANTE (Real): Nossa análise indica com alta certeza que este é um conteúdo de notícia real."
            elif confidence > 0.75:
                message = "MODELO CONFIANTE (Real): A classificação sugere que esta é uma notícia real, mas a margem de erro não é nula."
            else: 
                message = "MODELO INSEGURO (Pende para Real): A classificação pende levemente para real, mas com baixa confiança. Requer verificação humana."

        else: # pred == 0 (Modelo classifica como Fake News)
            if confidence > 0.90:
                message = "MODELO MUITO CONFIANTE (Falsa): Há forte indicação de que se trata de conteúdo enganoso ou falso. Classificado como Falsa."
            elif confidence > 0.75:
                message = "MODELO CONFIANTE (Falsa): A classificação aponta com razoável certeza que as informações são imprecisas ou falsas."
            else: 
                message = "MODELO INSEGURO (Pende para Falsa): A classificação pende levemente para falsa, mas com baixa confiança. Requer verificação humana."

        label = "Fake News" if pred == 0 else "Notícia Real"

        return {
            "prediction": label, 
            "confidence": round(confidence, 2),
            "message": message
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar: {str(e)}")

# Para desenvolvimento local
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)