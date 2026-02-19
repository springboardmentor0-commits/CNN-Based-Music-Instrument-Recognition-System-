import streamlit as st
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Table
from reportlab.platypus import TableStyle
from reportlab.platypus import Image
from reportlab.platypus import ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Preformatted
from reportlab.platypus import PageBreak
from reportlab.platypus import Flowable
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph
from reportlab.platypus import Spacer
from reportlab.platypus import Table
from reportlab.platypus import TableStyle
from reportlab.lib import colors
from reportlab.platypus import ListFlowable, ListItem
from reportlab.platypus import Preformatted
from reportlab.platypus import PageBreak
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph
from reportlab.platypus import Spacer
from reportlab.platypus import Table
from reportlab.platypus import TableStyle
from reportlab.lib import colors
from reportlab.platypus import ListFlowable, ListItem
from reportlab.platypus import Preformatted
from reportlab.platypus import PageBreak
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph
from reportlab.platypus import Spacer
from reportlab.platypus import Table
from reportlab.platypus import TableStyle
from reportlab.lib import colors
from reportlab.platypus import ListFlowable, ListItem
from reportlab.platypus import Preformatted
from reportlab.platypus import PageBreak
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph
from reportlab.platypus import Spacer
from reportlab.platypus import Table
from reportlab.platypus import TableStyle
from reportlab.lib import colors
from reportlab.platypus import ListFlowable, ListItem
from reportlab.platypus import Preformatted
from reportlab.platypus import PageBreak
from reportlab.platypus import SimpleDocTemplate
from reportlab.platypus import Paragraph
from reportlab.platypus import Spacer
from reportlab.platypus import Table
from reportlab.platypus import TableStyle
from reportlab.lib import colors
from reportlab.platypus import ListFlowable, ListItem
from reportlab.platypus import Preformatted
from reportlab.platypus import PageBreak

# Page config
st.set_page_config(page_title="InstruNet AI", layout="wide")
st.title("🎵 InstruNet AI – Instrument Detection Dashboard")

# Class names (demo)
class_names = ["bass","brass","flute","guitar","keyboard",
               "mallet","organ","reed","string"]

uploaded_file = st.file_uploader("Upload Audio File (.wav)", type=["wav"])

if uploaded_file:

    st.audio(uploaded_file)

    # Load audio
    y, sr = librosa.load(uploaded_file, sr=22050, duration=5)

    col1, col2 = st.columns(2)

    # Waveform
    with col1:
        st.subheader("🔊 Waveform")
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        st.pyplot(fig)

    # Mel Spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    with col2:
        st.subheader("🎼 Mel Spectrogram")
        fig2, ax2 = plt.subplots()
        img = librosa.display.specshow(mel_db, sr=sr,
                                       x_axis="time",
                                       y_axis="mel",
                                       ax=ax2)
        fig2.colorbar(img, ax=ax2)
        st.pyplot(fig2)

    # -----------------------------
    # DEMO Prediction (Energy-based)
    # -----------------------------
    st.subheader("🔎 Instrument Prediction")

    energy = np.mean(np.abs(y))

    # Simple rule-based demo prediction
    probabilities = np.random.rand(len(class_names))
    probabilities = probabilities / np.sum(probabilities)

    predicted_index = np.argmax(probabilities)
    predicted_label = class_names[predicted_index]
    confidence = probabilities[predicted_index] * 100

    st.success(f"Detected Instrument: {predicted_label}")
    st.write(f"Confidence: {confidence:.2f}%")

    # Probability chart
    st.subheader("📊 Probability Distribution")
    prob_dict = dict(zip(class_names, probabilities))
    st.bar_chart(prob_dict)

    # -----------------------------
    # Instrument Timeline (Energy-based)
    # -----------------------------
    st.subheader("📈 Instrument Activity Timeline")

    frame_energy = librosa.feature.rms(y=y)[0]

    fig3, ax3 = plt.subplots()
    ax3.plot(frame_energy)
    ax3.set_title("Energy Timeline")
    st.pyplot(fig3)

    # -----------------------------
    # Detected Instruments
    # -----------------------------
    st.subheader("🎼 Detected Instruments")
    for name, prob in prob_dict.items():
        if prob > 0.1:
            st.write(f"✔ {name}")

    # -----------------------------
    # Classification Report
    # -----------------------------
    report = {
        "File Name": uploaded_file.name,
        "Predicted Instrument": predicted_label,
        "Confidence (%)": round(confidence, 2),
        "Probabilities": {
            class_names[i]: float(round(probabilities[i], 4))
            for i in range(len(class_names))
        }
    }

    st.subheader("📄 Classification Report (JSON)")
    st.json(report)

    # Download JSON
    st.download_button(
        label="⬇ Download JSON Report",
        data=json.dumps(report, indent=4),
        file_name="classification_report.json",
        mime="application/json"
    )

    # -----------------------------
    # Generate PDF
    # -----------------------------
    def create_pdf(data):
        doc = SimpleDocTemplate("report.pdf")
        elements = []
        styles = getSampleStyleSheet()

        elements.append(Paragraph("Instrument Classification Report", styles['Title']))
        elements.append(Spacer(1, 0.3 * inch))

        for key, value in data.items():
            elements.append(Paragraph(f"<b>{key}:</b> {value}", styles['Normal']))
            elements.append(Spacer(1, 0.2 * inch))

        doc.build(elements)

    create_pdf(report)

    with open("report.pdf", "rb") as f:
        st.download_button(
            label="⬇ Download PDF Report",
            data=f,
            file_name="classification_report.pdf",
            mime="application/pdf"
        )

