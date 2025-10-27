from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
import numpy as np

from utils.preprocess import preprocess_text

model = load("model/model.pkl")

# uvicorn main:app --reload 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict(input: TextInput):
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

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("main:app", host="192.168.0.103", port=8000, reload=True)