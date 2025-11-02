import torch
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModel
from api.utils.preprocess import load_data

# Configurando modelo BERT
MODEL_NAME = "neuralmind/bert-base-portuguese-cased"
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"Usando dispositivo: {device}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
bert_model = AutoModel.from_pretrained(MODEL_NAME).to(device)

# Embeddings BERT
def get_bert_embedding(text):
    """Gera embedding médio do BERT para o texto."""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    ).to(device)

    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Média das representações do último hidden state
    embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
    return embedding.squeeze()

# Carregar dados
df = load_data("Project/data/fake_news/financeiros", "Project/data/real_news/financeiros")

print(f"Total de amostras: {len(df)}")

# Gerar embeddings
print("Gerando embeddings com BERTimbau...")
X = np.vstack([get_bert_embedding(text) for text in df["texto"]])
y = df["label"].values

# Treino do SVM
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Treinando SVM...")
svm = LinearSVC()
svm.fit(X_train, y_train)

# Avaliação
y_pred = svm.predict(X_test)
print("\n===== RESULTADOS =====")
print("Acurácia:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Salvar modelo
dump(svm, "Project/api/model/model.pkl")
print("Modelo salvo em Project/api/model/model.pkl")
