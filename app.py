import streamlit as st
import torch
import joblib
from transformers import DistilBertTokenizer, DistilBertModel

# --------------------------------------------------
# Page Setup
# --------------------------------------------------
st.set_page_config(page_title="User Profile Analysis", page_icon="📝")
st.title("📝 User Profile Analysis")
st.write("Fill in the profile details below.")

# --------------------------------------------------
# Load Models (Cached, one resource per step)
# --------------------------------------------------
@st.cache_resource
def load_tokenizer():
    return DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

@st.cache_resource
def load_bert():
    bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
    bert.eval()
    return bert

@st.cache_resource
def load_classifier():
    return joblib.load("models/HLP_potential_model.joblib")

def load_models_with_progress():
    """Show a detailed loading screen while the models are being prepared."""
    with st.status("Preparing the profile analyzer…", expanded=True) as status:
        st.write("📥 Loading the DistilBERT tokenizer…")
        tokenizer = load_tokenizer()

        st.write("🧠 Loading the DistilBERT language model — "
                 "the first run downloads it (~260 MB), so this can take a moment…")
        bert = load_bert()

        st.write("🎯 Loading the high-level-position classifier…")
        clf = load_classifier()

        status.update(label="✅ Models ready — fill in the profile below.",
                      state="complete", expanded=False)
    return tokenizer, bert, clf

# Show the full loading screen only on the first load. Streamlit reruns the
# script on every interaction, but the cached loaders return instantly after
# the first time, so there's no need to re-render the screen.
if not st.session_state.get("models_loaded"):
    tokenizer, bert_model, clf_model = load_models_with_progress()
    st.session_state.models_loaded = True
else:
    tokenizer = load_tokenizer()
    bert_model = load_bert()
    clf_model = load_classifier()

# --------------------------------------------------
# Embedding Helper
# --------------------------------------------------
def get_embedding(text):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = bert_model(**inputs)
    return outputs.last_hidden_state[:, 0, :].numpy()

# --------------------------------------------------
# Session State for Experiences
# --------------------------------------------------
if "experiences" not in st.session_state:
    st.session_state.experiences = [
        {"title": "", "duration": "", "description": ""}
    ]

def add_experience():
    st.session_state.experiences.append(
        {"title": "", "duration": "", "description": ""}
    )

# --------------------------------------------------
# Add Experience Button (Outside Form)
# --------------------------------------------------
st.subheader("Experience")
st.button("➕ Add another experience", on_click=add_experience)

# --------------------------------------------------
# Profile Form
# --------------------------------------------------
with st.form("profile_form"):

    about_text = st.text_area(
        "About Section",
        placeholder="Brief professional summary...",
        height=150
    )

    for idx, exp in enumerate(st.session_state.experiences):
        st.markdown(f"**Experience {idx + 1}**")

        exp["title"] = st.text_input(
            "Title",
            key=f"title_{idx}",
            value=exp["title"]
        )

        exp["duration"] = st.text_input(
            "Duration",
            key=f"duration_{idx}",
            value=exp["duration"]
        )

        exp["description"] = st.text_area(
            "Description",
            key=f"description_{idx}",
            value=exp["description"],
            height=100
        )

        st.divider()

    submitted = st.form_submit_button("Submit Profile", type="primary")

# --------------------------------------------------
# Handle Submission
# --------------------------------------------------
if submitted:

    if not about_text:
        st.error("Please fill in the About section.")
    else:
        experience_text = []
        for exp in st.session_state.experiences:
            if exp["title"] or exp["description"]:
                experience_text.append(
                    f"{exp['title']} ({exp['duration']}): {exp['description']}"
                )

        full_text = about_text + " " + " ".join(experience_text)

        with st.spinner("Analyzing profile..."):
            embedding = get_embedding(full_text)
            probabilities = clf_model.predict_proba(embedding)[0]
            high_level_prob = probabilities[1] * 100

        st.success("Analysis Complete")

        st.subheader("🎯 Results")

        with st.container(border=True):
            col1, col2 = st.columns([1, 3])

            with col1:
                st.metric(
                    label="High-Level Position Probability",
                    value=f"{high_level_prob:.1f}%"
                )

            with col2:
                if high_level_prob >= 70:
                    explanation = "Strong indicators of a high-level professional role."
                elif high_level_prob >= 40:
                    explanation = "Moderate indicators of a high-level professional role."
                else:
                    explanation = "Low indicators of a high-level professional role."

                st.info(f"**Insight:** {explanation}")
