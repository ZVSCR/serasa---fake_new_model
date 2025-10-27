import re
import string
import pandas as pd

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer

import nltk
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("rslp", quiet=True)

def preprocess_text(text: str) -> str:
    if not isinstance(text, str) or pd.isna(text):
        return ""

    text = text.lower()
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    try:
        tokens = word_tokenize(text, language="portuguese")
    except:
        tokens = text.split()

    stop_words = set(stopwords.words("portuguese"))
    tokens = [w for w in tokens if w not in stop_words and len(w) > 2]

    stemmer = RSLPStemmer()
    tokens = [stemmer.stem(w) for w in tokens]

    return " ".join(tokens)


def load_data(fake_news_dir: str, real_news_dir: str) -> pd.DataFrame:
    import os

    if not os.path.exists(fake_news_dir) or not os.path.exists(real_news_dir):
        raise FileNotFoundError("Problema no caminho dos diretorios.")

    data = []

    def load_files(directory, label):
        for filename in os.listdir(directory):
            if filename.endswith(".txt"):
                try:
                    with open(os.path.join(directory, filename), "r", encoding="utf-8") as file:
                        content = file.read()
                        data.append({"texto": content, "label": label})
                except Exception as e:
                    print(f"Erro ao ler {filename}: {e}")

    load_files(fake_news_dir, 0) 
    load_files(real_news_dir, 1) 

    if not data:
        raise ValueError("Nenhum arquivo de texto encontrado.")

    return pd.DataFrame(data)
