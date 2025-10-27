from fastapi import FastAPI, HTTPException
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

def get_model():
    global _model
    if _model is not None:
        return _model
    
    model_path = os.path.join(os.path.dirname(__file__), "model", "model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}")
    
    print(f"Carregando modelo de: {model_path}")
    _model = load(model_path)
    return _model


@app.get("/")
async def root():
    return {"message": "API de Detecção de Fake News"}

@app.get("/health")
async def health_check():
    try:
        model = get_model()
        return {"status": "healthy", "model_loaded": True}
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@app.post("/predict", response_model=PredictionOutput)
async def predict(input: TextInput):
    try:
        model = get_model()
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
        if confidence > 0.9:
            message = "Essa notícia parece bastante confiável."
        elif confidence > 0.75:
            message = "Parece ser verdadeira, mas é sempre bom conferir as fontes."
        elif confidence > 0.6:
            message = "Cuidado — há indícios de inconsistência, verifique outras fontes."
        elif confidence > 0.4:
            message = "Atenção! Essa notícia pode conter informações imprecisas."
        else:
            message = "Alerta: fortes indícios de que essa notícia é falsa."

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