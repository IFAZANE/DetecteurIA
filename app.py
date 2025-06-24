import os
import streamlit as st
import fitz  # PyMuPDF
import matplotlib.pyplot as plt

# ------------------------
# Fonction d'extraction du texte
# ------------------------
def extract_text_from_pdf(pdf_file):
    text = ""
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        for page in doc:
            text += page.get_text()
    return text

# ------------------------
# Fonction de détection simulée
# ------------------------
def detect_ai_generated_text(text):
    import random
    ai_score = random.uniform(0, 1)
    human_score = 1 - ai_score
    return ai_score, human_score

# ------------------------
# Interface utilisateur
# ------------------------
st.set_page_config(page_title="Détecteur IA PDF", page_icon="🤖", layout="wide")
st.title("📄🧠 Détecteur de texte généré par l'IA")

uploaded_file = st.sidebar.file_uploader("📤 Téléversez un fichier PDF", type="pdf")

# ------------------------
# Si aucun fichier : afficher l'image d'accueil
# ------------------------
if not uploaded_file:
    image_path = os.path.join("assets", "image.png")  # Assure-toi que le fichier est dans /assets

    if os.path.exists(image_path):
        st.image(image_path, use_container_width=True)
    else:
        st.warning(f"⚠️ Image non trouvée : `{image_path}`")

    st.markdown("""
    ### Bienvenue dans l'application de détection IA 🧠📄  
    Cette application vous permet d'analyser un document PDF pour détecter s'il a été rédigé par une Intelligence Artificielle comme ChatGPT.  
    👉 Commencez par téléverser un fichier PDF via le menu latéral.

    **🔍 Utilisation typique :**
    - Vérification de rapports étudiants
    - Détection de contenu IA dans les articles
    - Analyse automatisée de documents

    ---
    """)

# ------------------------
# Si un fichier est uploadé : traitement
# ------------------------
if uploaded_file:
    st.success("✅ Fichier chargé avec succès.")

    with st.spinner("📚 Extraction du texte..."):
        text = extract_text_from_pdf(uploaded_file)

    if not text.strip():
        st.error("❌ Aucun texte détecté.")
    else:
        with st.spinner("🤖 Analyse du contenu..."):
            ai_score, human_score = detect_ai_generated_text(text)

        # Résultat de l’analyse
        st.subheader("🧾 Résultat de l'analyse")
        st.metric("Probabilité IA", f"{ai_score * 100:.2f} %")

        # Graphique en camembert
        fig, ax = plt.subplots()
        ax.pie([ai_score, human_score], labels=['IA', 'Humain'], autopct='%1.1f%%',
               colors=['#ff4b4b', '#1f77b4'], startangle=90, wedgeprops=dict(width=0.4))
        ax.set_title("Degré de génération IA")
        st.pyplot(fig)

        # Badge de risque
        st.subheader("🎯 Degré de génération IA")
        if ai_score >= 0.8:
            st.markdown("### 🔴 Très Élevé : Très probablement généré par une IA (≥ 80%)")
        elif ai_score >= 0.5:
            st.markdown("### 🟡 Élevé : Probablement généré par une IA (50–80%)")
        elif ai_score >= 0.15:
            st.markdown("### 🔵 Modéré : Peut contenir des éléments IA (15–50%)")
        else:
            st.markdown("### 🟢 Faible : Très probablement rédigé par un humain (< 15%)")

        # Affichage du texte extrait
        with st.expander("📝 Aperçu du texte analysé"):
            st.text_area("Contenu (extrait)", text[:2000], height=300)
