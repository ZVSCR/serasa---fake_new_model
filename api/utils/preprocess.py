import re
import unicodedata
import pandas as pd

STOPWORDS_PT = {
    "a", "à", "ao", "aos", "as", "às", "de", "da", "das", "do", "dos", "e",
    "em", "no", "na", "nas", "nos", "num", "numa", "nuns", "numas",
    "por", "para", "com", "como", "que", "o", "os", "uma", "umas", "um", "uns",
    "se", "sua", "suas", "seu", "seus", "é", "ser", "foi", "era", "são", "será",
    "ao", "aos", "esta", "este", "isto", "isso", "aquele", "aquela", "aquilo",
    "há", "houve", "ter", "têm", "tem", "havia", "tinha", "tinham", "tido",
    "já", "não", "mas", "ou", "porque", "quando", "onde", "então", "também",
    "muito", "pouco", "mesmo", "outro", "outra", "outros", "outras",
    "sobre", "entre", "após", "antes", "até", "desde", "sem", "sob",
    "meu", "minha", "meus", "minhas", "teu", "tua", "teus", "tuas", "dele", "dela",
    "deles", "delas", "nosso", "nossa", "nossos", "nossas"
}


def normalize_accents(text: str) -> str:
    """Remove acentuação e caracteres não-ASCII."""
    text = unicodedata.normalize("NFKD", text)
    return "".join([c for c in text if not unicodedata.combining(c)])


def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""

    text = normalize_accents(text.lower())

    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"[^\w\s]", " ", text)

    tokens = text.split()

    tokens = [t for t in tokens if len(t) > 2 and t not in STOPWORDS_PT]

    def light_stem(word):
        for suf in ("mente", "ções", "sões", "ção", "são", "mente", "dade", "dades", "ismo", "ismos", "ista", "istas"):
            if word.endswith(suf):
                return word[:-len(suf)]
        return word

    tokens = [light_stem(t) for t in tokens]

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