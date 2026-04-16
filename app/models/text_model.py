import os
from pathlib import Path
import joblib
from sklearn.pipeline import Pipeline

import re
import spacy
import numpy as np

TEXT_MODEL_NAME = "best_model_annonces.pkl"
TEXT_class_names = ["Sans défauts", "Avec défauts"]

nlp = spacy.load('fr_core_news_sm')

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

def load_text_model() -> Pipeline:
    project_root = Path.cwd()
    print("project_root : " + str(project_root))
    TEXT_MODEL_PATH = "." + str(project_root) + "/" + TEXT_MODEL_NAME
    print("TEXT_MODEL_PATH : ", TEXT_MODEL_PATH)
    if not os.path.exists(TEXT_MODEL_PATH):
        print("*** Aucun modèle image trouvé !!!")
        text_model = None
    else: 
        text_model = joblib.load(TEXT_MODEL_PATH)
        print("Modèle texte chargé.")
    return text_model

def predict_text(model: Pipeline, text: str) -> dict:
    print("predict_text")
    print("text : ", text)
    cleaned = preprocess_text(text)
    print("cleaned : ", cleaned)

    # Prédiction
    pred = model.predict([cleaned])[0]
    print("pred : ", str(pred))

    # Probabilités complètes
    proba = model.predict_proba([cleaned])[0]
    print("proba : ", str(proba))

    proba_0 = proba[0]
    proba_1 = proba[1]

    print(f"Prédiction : {pred} (1=défaut, 0=ok)")
    print(f"Probabilité PAS de défaut (0) : {proba_0:.2f}")
    print(f"Probabilité défaut (1) : {proba_1:.2f}")

    digit = int(np.argmax(proba))
    class_name = TEXT_class_names[digit] 
    confidence = round(float(np.max(proba)), 4)
    probabilities = {
        TEXT_class_names[i]: round(float(proba[i]), 4) for i in range(len(TEXT_class_names))
    }

    return {"class_name": class_name, "confidence": confidence, "probabilities": probabilities}