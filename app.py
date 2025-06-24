import os
import re
import streamlit as st
import fitz  # PyMuPDF
import matplotlib.pyplot as plt
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# ------------------------
# Résumé automatique (modèle plus léger)
# ------------------------

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-3")  # version + légère

def summarize_text(text, max_tokens=1024):
    summarizer = load_summarizer()
    # Limiter à 1 ou 2 chunks max pour éviter surcharge CPU/mémoire
    chunks = [text[i:i+1000] for i in range(0, min(len(text), 2000), 1000)]
    summary = ""
    for chunk in chunks:
        res = summarizer(chunk, max_length=100, min_length=30, do_sample=False)
        summary += res[0]['summary_text'] + " "
    return summary.strip()

# ------------------------
# Extraction des données chiffrées
# ------------------------

def extract_numbers(text):
    pattern = r"\b(?:\d{1,3}(?:[\.,]\d{3})*|\d+)(?:[%€$]?| [A-Za-z]*)\b"
    matches = re.findall(pattern, text)
    return sorted(set(matches), key=lambda x: text.find(x))

# ------------------------
# Détection IA avec Roberta (utiliser 1 seul modèle en mémoire)
# ------------------------

@st.cache_resource
def load_ai_detector():
    tokenizer = AutoTokenizer.from_pretrained("roberta-base-openai-detector")
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base-openai-detector")
    model.eval()
    return tokenizer, model

def detect_ai_generated_text(text):
    tokenizer, model = load_ai_detector()
    short_text = text[:1000]  # Réduire la longueur du texte pour le modèle
    inputs = tokenizer(short_text, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1).squeeze()
    return float(probs[1]), float(probs[0])

# ------------------------
# Extraction du texte PDF
# ------------------------

def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
            if len(text) > 3000:
                break  # Limiter l'extraction à 3000 caractères max
    return text

# ------------------------
# Interface utilisateur
# ------------------------

st.set_page_config(page_title="Détecteur IA PDF", page_icon="🤖", layout="wide")
st.title("📄🧠 Détecteur de texte généré par l'IA")

uploaded_file = st.sidebar.file_uploader("📤 Téléversez un fichier PDF", type="pdf")

if not uploaded_file:
    image_path = os.path.join("assets", "image.png")
    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning("⚠️ Image d'accueil non trouvée.")
    st.markdown("""
    ### Bienvenue dans l'application de détection IA 🧠📄  
    👉 Téléversez un fichier PDF via le menu latéral pour commencer.
    """)

if uploaded_file:
    st.success("✅ Fichier chargé avec succès.")

    with st.spinner("📚 Extraction du texte..."):
        text = extract_text_from_pdf(uploaded_file)

    if not text.strip():
        st.error("❌ Aucun texte détecté.")
    else:
        with st.spinner("🤖 Analyse du contenu..."):
            ai_score, human_score = detect_ai_generated_text(text)

        st.subheader("🧾 Résultat de l'analyse")
        st.metric("Probabilité IA", f"{ai_score * 100:.2f} %")

        fig, ax = plt.subplots()
        ax.pie([ai_score, human_score], labels=['IA', 'Humain'], autopct='%1.1f%%',
               colors=['#ff4b4b', '#1f77b4'], startangle=90, wedgeprops=dict(width=0.4))
        ax.set_title("Degré de génération IA")
        st.pyplot(fig)

        st.subheader("🎯 Interprétation")
        if ai_score >= 0.8:
            st.markdown("### 🔴 Très Élevé : Très probablement généré par une IA (≥ 80%)")
        elif ai_score >= 0.5:
            st.markdown("### 🟡 Élevé : Probablement généré par une IA (50–80%)")
        elif ai_score >= 0.15:
            st.markdown("### 🔵 Modéré : Peut contenir des éléments IA (15–50%)")
        else:
            st.markdown("### 🟢 Faible : Très probablement rédigé par un humain (< 15%)")

        with st.expander("🧠 Résumé automatique du texte"):
            with st.spinner("✍️ Résumé en cours..."):
                summary = summarize_text(text)
            st.write(summary)

        with st.expander("📊 Données clés détectées"):
            numbers = extract_numbers(text)
            if numbers:
                st.write(", ".join(numbers))
            else:
                st.info("Aucune donnée chiffrée trouvée.")

        with st.expander("📝 Aperçu du texte analysé"):
            st.text_area("Contenu (extrait)", text[:1500], height=300)
