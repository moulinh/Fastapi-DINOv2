import re
import string
import spacy

nlp = spacy.load('fr_core_news_sm')

"""
def preprocess_text(text: str) -> str:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text
"""

def preprocess_text(text: str) -> str:
    print("preprocess_text")
    print("text : ", text)
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Zàâéèêôùûç\s]', '', text)
    doc = nlp(text)
    lemmatized = ' '.join([token.lemma_ for token in doc if not token.is_stop])
    print("fin preprocess_text")
    print("retour : ", lemmatized)
    return str(lemmatized)
