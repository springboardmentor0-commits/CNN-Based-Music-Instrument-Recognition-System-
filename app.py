import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import json
import seaborn as sns
from tensorflow.keras.models import load_model

st.set_page_config(page_title="InstruNet AI", layout="wide")

# ================= ADVANCED ANIMATED UI =================
st.markdown("""
<style>

/* Animated Gradient Background */
.stApp {
    background: linear-gradient(-45deg, #1f4037, #99f2c8, #4e73df, #1cc88a);
    background-size: 400% 400%;
    animation: gradientBG 12s ease infinite;
    color: white;
}

/* Gradient Animation */
@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* Glassmorphism Main Container */
.block-container {
    background: rgba(255, 255, 255, 0.15);
    padding: 2rem;
    border-radius: 20px;
    backdrop-filter: blur(12px);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    animation: fadeIn 1.5s ease-in-out;
}

/* Fade In */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

/* Title Styling */
h1 {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    animation: fadeIn 2s ease-in-out;
}

/* Floating Music Notes */
.music-note {
    position: fixed;
    font-size: 28px;
    animation: float 10s linear infinite;
    opacity: 0.5;
}

.music1 { left: 15%; animation-delay: 0s; }
.music2 { left: 50%; animation-delay: 3s; }
.music3 { left: 80%; animation-delay: 6s; }

@keyframes float {
    0% { bottom: -10%; }
    100% { bottom: 110%; }
}

/* Prediction Card */
.prediction-card {
    padding: 25px;
    border-radius: 20px;
    background: linear-gradient(to right, #ff416c, #ff4b2b);
    color: white;
    font-size: 22px;
    text-align: center;
    box-shadow: 0 0 30px rgba(0,0,0,0.5);
    animation: fadeIn 2s ease-in-out;
}

</style>

<div class="music-note music1">🎵</div>
<div class="music-note music2">🎶</div>
<div class="music-note music3">🎼</div>

""", unsafe_allow_html=True)

st.title("🎵 InstruNet AI - CNN Based Instrument Detection")

# ---------------- LOAD MODEL ----------------
@st.cache_resource
def load_cnn_model():
    return load_model("instrunet_final.keras")

model = load_cnn_model()

# ---------------- LOAD LABEL MAP ----------------
with open("label_map.json", "r") as f:
    label_map = json.load(f)

inv_label_map = {v: k for k, v in label_map.items()}
class_names = [inv_label_map[i] for i in range(len(inv_label_map))]

# ---------------- PREPROCESS FUNCTION ----------------
def preprocess_audio(audio_path, sr=16000):

    signal, _ = librosa.load(audio_path, sr=sr, mono=True)

    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    if log_mel.shape[1] < 128:
        pad_width = 128 - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)))
    else:
        log_mel = log_mel[:, :128]

    denominator = log_mel.max() - log_mel.min()
    if denominator != 0:
        log_mel = (log_mel - log_mel.min()) / denominator
    else:
        log_mel = np.zeros_like(log_mel)

    return log_mel.reshape(1, 128, 128, 1)

# ---------------- SEGMENT WISE FUNCTION ----------------
def segment_wise_prediction(audio_path, sr=16000):

    signal, _ = librosa.load(audio_path, sr=sr, mono=True)

    segment_length = sr
    total_segments = len(signal) // segment_length
    timeline_results = []

    for i in range(total_segments):
        segment = signal[i * segment_length:(i + 1) * segment_length]

        mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=128)
        log_mel = librosa.power_to_db(mel, ref=np.max)

        if log_mel.shape[1] < 128:
            pad_width = 128 - log_mel.shape[1]
            log_mel = np.pad(log_mel, ((0, 0), (0, pad_width)))
        else:
            log_mel = log_mel[:, :128]

        denominator = log_mel.max() - log_mel.min()
        if denominator != 0:
            log_mel = (log_mel - log_mel.min()) / denominator
        else:
            log_mel = np.zeros_like(log_mel)

        log_mel = log_mel.reshape(1, 128, 128, 1)

        prediction = model.predict(log_mel)[0]
        timeline_results.append(prediction)

    return np.array(timeline_results)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])

if uploaded_file:

    import tempfile

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    st.audio(tmp_path)
    st.markdown("---")

    # ================= WAVEFORM =================
    st.subheader("🎵 Amplitude vs Time (Waveform)")
    signal, sr = librosa.load(tmp_path, sr=16000)

    fig_wave, ax_wave = plt.subplots(figsize=(10, 4))
    librosa.display.waveshow(signal, sr=sr, ax=ax_wave)
    ax_wave.set_xlabel("Time (seconds)")
    ax_wave.set_ylabel("Amplitude")
    st.pyplot(fig_wave)

    st.markdown("---")

    # ================= MEL SPECTROGRAM =================
    st.subheader("🎼 Mel Spectrogram")

    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=128)
    log_mel = librosa.power_to_db(mel, ref=np.max)

    fig_spec, ax_spec = plt.subplots(figsize=(10, 4))
    img = librosa.display.specshow(
        log_mel,
        sr=sr,
        x_axis='time',
        y_axis='mel',
        ax=ax_spec
    )
    fig_spec.colorbar(img, ax=ax_spec, format='%+2.0f dB')
    st.pyplot(fig_spec)

    st.markdown("---")

    # ================= PREDICTION =================
    with st.spinner("🔍 Analyzing Audio..."):
        X_input = preprocess_audio(tmp_path)
        prediction = model.predict(X_input)[0]

    dominant_idx = np.argmax(prediction)
    dominant_instrument = class_names[dominant_idx]
    dominant_conf = round(prediction[dominant_idx] * 100, 2)

    st.markdown(f"""
    <div class="prediction-card">
        🎯 Predicted Instrument: <b>{dominant_instrument.upper()}</b><br>
        Confidence: <b>{dominant_conf}%</b>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ================= CONFIDENCE CHART =================
    st.subheader("📊 Instrument Confidence Levels")

    fig_conf, ax_conf = plt.subplots()
    ax_conf.barh(class_names, prediction)
    ax_conf.set_xlim(0, 1)
    ax_conf.set_xlabel("Probability")
    st.pyplot(fig_conf)

    st.markdown("---")

    # ================= TIMELINE =================
    st.subheader("🕒 Instrument Activity Timeline")
    timeline_preds = segment_wise_prediction(tmp_path)

    if timeline_preds.shape[0] > 0:
        fig2, ax2 = plt.subplots(figsize=(10, 5))
        sns.heatmap(
            timeline_preds.T,
            cmap="viridis",
            xticklabels=[f"{i}-{i+1}s" for i in range(timeline_preds.shape[0])],
            yticklabels=class_names,
            ax=ax2
        )
        st.pyplot(fig2)
    else:
        st.warning("Audio too short for segment-wise analysis.")
