from ml.embedding import bert_embedding

LABEL_MAP = {
    0: "Non High-Level Position",
    1: "High-Level Leadership Potential"
}

def predict_profile(text, tokenizer, bert_model, classifier):
    embedding = bert_embedding(text, tokenizer, bert_model)
    prediction = classifier.predict(embedding)[0]
    probability = classifier.predict_proba(embedding)[0][prediction]

    return {
        "label_id": prediction,
        "label": LABEL_MAP[prediction],
        "confidence": probability
    }
