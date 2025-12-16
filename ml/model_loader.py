import joblib
import streamlit as st
from transformers import DistilBertTokenizer, DistilBertModel

@st.cache_resource
def load_models():
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased"
    )
    bert_model = DistilBertModel.from_pretrained(
        "distilbert-base-uncased"
    )
    classifier = joblib.load(
        "models/HLP_potential_model.joblib"
    )
    return tokenizer, bert_model, classifier
