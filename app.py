import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import json
from fpdf import FPDF
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="InstruNet AI: Music Instrument Recognition",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Advanced CSS - Premium UI Design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    * {
        font-family: 'Poppins', 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    /* Animated gradient background */
    .stApp {
        background: linear-gradient(135deg, #0a0a0a 0%, #1a1a2e 50%, #0a0a0a 100%) !important;
        background-size: 400% 400% !important;
        animation: gradientShift 15s ease infinite !important;
    }
    
    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    .main {
        background-color: transparent !important;
        color: #ffffff;
    }
    
    /* Container with better spacing */
    .block-container {
        padding: 3rem 4rem !important;
        max-width: 100% !important;
    }
    
    /* Glowing title with text gradient */
    h1 {
        background: linear-gradient(135deg, #00d4ff 0%, #2196F3 50%, #00ffaa 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
        background-clip: text !important;
        font-weight: 800 !important;
        font-size: 48px !important;
        margin-bottom: 10px !important;
        text-align: center !important;
        letter-spacing: -1px !important;
        text-shadow: 0 0 40px rgba(33, 150, 243, 0.5) !important;
    }
    
    /* Section headers with gradient underline */
    h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 22px !important;
        margin-bottom: 25px !important;
        padding-bottom: 12px !important;
        border-bottom: 3px solid transparent !important;
        border-image: linear-gradient(90deg, #2196F3, #00d4ff, #00ffaa) 1 !important;
        letter-spacing: 0.5px !important;
    }
    
    /* Glassmorphism cards */
    [data-testid="column"] {
        background: rgba(45, 55, 72, 0.6) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-radius: 20px !important;
        padding: 30px !important;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.4),
                    inset 0 1px 0 rgba(255, 255, 255, 0.1) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="column"]:hover {
        transform: translateY(-5px) !important;
        box-shadow: 0 12px 40px rgba(33, 150, 243, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.2) !important;
    }
    
    /* Premium file uploader */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.1) 0%, rgba(0, 212, 255, 0.1) 100%) !important;
        border: 2px dashed rgba(33, 150, 243, 0.5) !important;
        border-radius: 16px !important;
        padding: 40px !important;
        text-align: center !important;
        transition: all 0.3s ease !important;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #2196F3 !important;
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.15) 0%, rgba(0, 212, 255, 0.15) 100%) !important;
        box-shadow: 0 0 30px rgba(33, 150, 243, 0.3) !important;
    }
    
    [data-testid="stFileUploader"] section {
        border: none !important;
        background-color: transparent !important;
    }
    
    [data-testid="stFileUploader"] label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 18px !important;
    }
    
    /* Neon blue browse button */
    [data-testid="stFileUploader"] button {
        background: linear-gradient(135deg, #2196F3 0%, #00d4ff 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 14px 36px !important;
        font-weight: 700 !important;
        border: none !important;
        font-size: 15px !important;
        box-shadow: 0 0 20px rgba(33, 150, 243, 0.6),
                    0 5px 20px rgba(33, 150, 243, 0.4) !important;
        transition: all 0.4s ease !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
    }
    
    [data-testid="stFileUploader"] button:hover {
        background: linear-gradient(135deg, #00d4ff 0%, #00ffaa 100%) !important;
        box-shadow: 0 0 35px rgba(0, 212, 255, 0.8),
                    0 8px 30px rgba(33, 150, 243, 0.6) !important;
        transform: translateY(-3px) scale(1.05) !important;
    }
    
    /* ANALYZE TRACK - Glowing neon button */
    .stButton > button {
        background: linear-gradient(135deg, #2196F3 0%, #00d4ff 50%, #00ffaa 100%) !important;
        color: white !important;
        border-radius: 12px !important;
        padding: 16px 32px !important;
        font-weight: 800 !important;
        border: none !important;
        width: 100% !important;
        text-transform: uppercase !important;
        font-size: 15px !important;
        letter-spacing: 2px !important;
        box-shadow: 0 0 25px rgba(33, 150, 243, 0.7),
                    0 5px 25px rgba(33, 150, 243, 0.5) !important;
        transition: all 0.4s ease !important;
        position: relative !important;
        overflow: hidden !important;
    }
    
    .stButton > button::before {
        content: '' !important;
        position: absolute !important;
        top: 0 !important;
        left: -100% !important;
        width: 100% !important;
        height: 100% !important;
        background: linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent) !important;
        transition: left 0.5s !important;
    }
    
    .stButton > button:hover::before {
        left: 100% !important;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #00ffaa 0%, #00d4ff 50%, #2196F3 100%) !important;
        box-shadow: 0 0 40px rgba(0, 255, 170, 0.8),
                    0 8px 35px rgba(33, 150, 243, 0.7) !important;
        transform: translateY(-3px) scale(1.02) !important;
    }
    
    /* Export buttons with glow */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #2196F3 0%, #00d4ff 100%) !important;
        color: white !important;
        border-radius: 10px !important;
        padding: 12px 24px !important;
        font-weight: 700 !important;
        font-size: 12px !important;
        width: 100% !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        box-shadow: 0 0 15px rgba(33, 150, 243, 0.5),
                    0 4px 20px rgba(33, 150, 243, 0.3) !important;
        border: none !important;
        transition: all 0.3s ease !important;
    }
    
    .stDownloadButton > button:hover {
        background: linear-gradient(135deg, #00d4ff 0%, #00ffaa 100%) !important;
        box-shadow: 0 0 25px rgba(0, 212, 255, 0.7),
                    0 6px 25px rgba(33, 150, 243, 0.5) !important;
        transform: translateY(-2px) scale(1.03) !important;
    }
    
    /* Info boxes with neon border */
    .stAlert {
        background: linear-gradient(135deg, rgba(33, 150, 243, 0.15) 0%, rgba(0, 212, 255, 0.1) 100%) !important;
        border: 2px solid #2196F3 !important;
        border-radius: 16px !important;
        color: #64b5f6 !important;
        padding: 25px !important;
        font-weight: 500 !important;
        box-shadow: 0 0 20px rgba(33, 150, 243, 0.3),
                    inset 0 0 20px rgba(33, 150, 243, 0.1) !important;
        backdrop-filter: blur(10px) !important;
    }
    
    .stAlert p {
        color: #64b5f6 !important;
        margin: 0 !important;
        font-size: 15px !important;
    }
    
    /* Checkboxes with custom colors */
    .stCheckbox {
        color: #ffffff !important;
        padding: 10px 0 !important;
    }
    
    .stCheckbox label {
        color: #ffffff !important;
        font-weight: 600 !important;
        font-size: 15px !important;
    }
    
    .stCheckbox input[type="checkbox"]:checked {
        accent-color: #00ffaa !important;
        filter: drop-shadow(0 0 5px rgba(0, 255, 170, 0.6)) !important;
    }
    
    /* Stylized audio player */
    audio {
        width: 100% !important;
        margin: 20px 0 !important;
        border-radius: 12px !important;
        filter: drop-shadow(0 4px 15px rgba(33, 150, 243, 0.3)) !important;
    }
    
    /* Hide branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {visibility: hidden;}
    
    /* Enhanced text */
    small {
        color: #aaaaaa !important;
        font-size: 13px !important;
        font-weight: 400 !important;
    }
    
    /* Glowing charts */
    .js-plotly-plot {
        border-radius: 16px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* Markdown enhancements */
    .stMarkdown p {
        color: #e0e0e0 !important;
        line-height: 1.6 !important;
    }
    
    /* Subtitle with glow */
    .stMarkdown p[style*="center"] {
        color: #64b5f6 !important;
        text-shadow: 0 0 10px rgba(100, 181, 246, 0.5) !important;
        font-weight: 500 !important;
    }
    
    /* Dividers */
    hr {
        border: none !important;
        height: 2px !important;
        background: linear-gradient(90deg, transparent, #2196F3, transparent) !important;
        margin: 30px 0 !important;
    }
    
    /* Spinner overlay */
    .stSpinner > div {
        border-color: #2196F3 transparent #2196F3 transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'audio_file' not in st.session_state:
    st.session_state.audio_file = None
if 'analyzed' not in st.session_state:
    st.session_state.analyzed = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None

# Title
st.markdown("<h1 style='text-align: center;'>InstruNet AI: Music Instrument Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888888; font-size: 16px; margin-top: 5px;'>Upload. Analyze. Discover.</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Function to analyze audio
def analyze_audio(audio_file):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=22050)
    duration = librosa.get_duration(y=y, sr=sr)
    
    # Extract features
    spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)[0]
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)[0]
    
    # Segment for timeline (100 segments)
    segment_length = max(1, len(rms) // 100)
    
    instruments_detected = {}
    
    # Piano detection
    piano_score = min(100, int(np.mean(np.max(chroma, axis=0)) * 120))
    piano_timeline = []
    for i in range(100):
        start = i * segment_length
        end = min((i + 1) * segment_length, len(rms))
        if end > start:
            val = np.mean(np.max(chroma[:, start:end], axis=0)) * 150
            piano_timeline.append(min(100, int(val)))
        else:
            piano_timeline.append(0)
    
    instruments_detected['Piano'] = {
        'confidence': piano_score,
        'present': piano_score > 45,
        'timeline': np.array(piano_timeline)
    }
    
    # Drums detection
    drums_score = min(100, int(np.mean(zero_crossing_rate) * 50 + np.var(rms) * 20))
    drums_timeline = []
    for i in range(100):
        start = i * segment_length
        end = min((i + 1) * segment_length, len(rms))
        if end > start:
            val = np.mean(rms[start:end]) * 150
            drums_timeline.append(min(100, int(val)))
        else:
            drums_timeline.append(0)
    
    instruments_detected['Drums'] = {
        'confidence': drums_score,
        'present': drums_score > 35,
        'timeline': np.array(drums_timeline)
    }
    
    # Guitar detection
    guitar_score = min(100, int((np.mean(spectral_centroids) / sr) * 250))
    guitar_timeline = []
    for i in range(100):
        start = i * segment_length
        end = min((i + 1) * segment_length, len(spectral_centroids))
        if end > start:
            val = np.mean(spectral_centroids[start:end]) / sr * 350
            guitar_timeline.append(min(100, int(val)))
        else:
            guitar_timeline.append(0)
    
    instruments_detected['Guitar'] = {
        'confidence': guitar_score,
        'present': guitar_score > 30,
        'timeline': np.array(guitar_timeline)
    }
    
    # Bass detection
    bass_score = min(100, int((1 - np.mean(spectral_centroids) / sr) * 90))
    bass_timeline = []
    for i in range(100):
        start = i * segment_length
        end = min((i + 1) * segment_length, len(spectral_rolloff))
        if end > start:
            val = (1 - np.mean(spectral_rolloff[start:end]) / sr) * 120
            bass_timeline.append(min(100, max(0, int(val))))
        else:
            bass_timeline.append(0)
    
    instruments_detected['Bass'] = {
        'confidence': bass_score,
        'present': bass_score > 25,
        'timeline': np.array(bass_timeline)
    }
    
    # Saxophone detection
    sax_score = min(100, int((np.mean(spectral_rolloff) / sr) * 180))
    sax_timeline = []
    for i in range(100):
        start = i * segment_length
        end = min((i + 1) * segment_length, len(spectral_rolloff))
        if end > start:
            val = np.mean(spectral_rolloff[start:end]) / sr * 250
            sax_timeline.append(min(100, int(val)))
        else:
            sax_timeline.append(0)
    
    instruments_detected['Saxophone'] = {
        'confidence': sax_score,
        'present': sax_score > 45,
        'timeline': np.array(sax_timeline)
    }
    
    return y, sr, instruments_detected

# Function to create mini waveform
def create_waveform(y, sr):
    fig, ax = plt.subplots(figsize=(6, 1.2))
    fig.patch.set_facecolor('#2a2a2a')
    ax.set_facecolor('#2a2a2a')
    
    time = np.linspace(0, len(y) / sr, num=len(y))
    ax.plot(time, y, color='#2196F3', linewidth=0.6, alpha=0.8)
    ax.fill_between(time, y, color='#2196F3', alpha=0.3)
    ax.set_xlim([0, len(y) / sr])
    ax.set_ylim([-1, 1])
    ax.axis('off')
    plt.tight_layout(pad=0)
    return fig

# Function to create colorful spectrogram
def create_spectrogram(y, sr):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor('#2a2a2a')
    ax.set_facecolor('#1a1a1a')
    
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    img = librosa.display.specshow(D, y_axis='linear', x_axis='time', sr=sr, ax=ax, cmap='turbo')
    
    ax.set_xlabel('Intensity', color='#888888', fontsize=10)
    ax.set_ylabel('', color='white')
    ax.tick_params(colors='#888888', labelsize=8)
    ax.grid(True, alpha=0.15, color='#555555', linewidth=0.5)
    
    # Colorbar on the left
    cbar = plt.colorbar(img, ax=ax, pad=0.01)
    cbar.ax.tick_params(colors='#888888', labelsize=8)
    cbar.outline.set_edgecolor('#555555')
    cbar.outline.set_linewidth(0.5)
    
    plt.tight_layout()
    return fig

# Function to create instrument confidence bars
def create_confidence_bars(instruments):
    fig = go.Figure()
    
    instruments_list = []
    confidence_list = []
    colors_list = []
    
    # Vibrant colors for each instrument
    instrument_colors = {
        'Piano': '#2196F3',      # Blue
        'Drums': '#4CAF50',      # Green
        'Guitar': '#FF9800',     # Orange
        'Bass': '#9C27B0',       # Purple
        'Saxophone': '#F44336'   # Red
    }
    
    for instrument, data in instruments.items():
        instruments_list.append(instrument)
        confidence_list.append(data['confidence'])
        colors_list.append(instrument_colors.get(instrument, '#2196F3'))
    
    fig.add_trace(go.Bar(
        x=confidence_list,
        y=instruments_list,
        orientation='h',
        marker=dict(color=colors_list),
        text=[f"{c}%" for c in confidence_list],
        textposition='outside',
        textfont=dict(color='#888888', size=10)
    ))
    
    fig.update_layout(
        template='plotly_dark',
        height=260,
        showlegend=False,
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#2a2a2a',
        margin=dict(l=60, r=80, t=10, b=30),
        xaxis=dict(
            range=[0, 110],
            showgrid=True,
            gridcolor='#333333',
            tickvals=[0, 5, 10, 40],
            ticktext=['5 %', '5 %', '10 %', '40 %'],
            tickfont=dict(color='#888888', size=9),
            showline=False
        ),
        yaxis=dict(
            tickfont=dict(color='#ffffff', size=11),
            showline=False
        )
    )
    
    return fig

# Function to create instrument timeline
def create_timeline(instruments):
    fig = go.Figure()
    
    x_values = list(range(100))
    colors = {
        'Piano': '#2196F3',
        'Drums': '#4CAF50',
        'Guitar': '#FF9800',
        'Bass': '#9C27B0',
        'Saxophone': '#F44336'
    }
    
    for instrument, data in instruments.items():
        if data['present']:
            fig.add_trace(go.Scatter(
                x=x_values,
                y=data['timeline'],
                mode='lines',
                name=instrument,
                line=dict(width=2, color=colors.get(instrument, '#666666')),
                fill='tozeroy',
                fillcolor=colors.get(instrument, '#666666'),
                opacity=0.6,
                hovertemplate=f'{instrument}<br>Intensity: %{{y}}<extra></extra>'
            ))
    
    fig.update_layout(
        title=dict(text='Instrument Timeline', font=dict(color='white', size=14, family='Inter')),
        xaxis_title='',
        yaxis_title='',
        template='plotly_dark',
        height=220,
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='top',
            y=1.15,
            xanchor='left',
            x=0,
            font=dict(color='white', size=10),
            bgcolor='rgba(0,0,0,0)'
        ),
        plot_bgcolor='#1a1a1a',
        paper_bgcolor='#2a2a2a',
        margin=dict(l=30, r=20, t=50, b=30),
        xaxis=dict(
            showgrid=False,
            tickvals=[0, 25, 50, 75, 100],
            tickfont=dict(size=8, color='#666666'),
            showline=False
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#333333',
            tickvals=[0, 25, 50, 75, 100],
            tickfont=dict(size=8, color='#666666'),
            showline=False
        )
    )
    
    return fig

# Export functions
def export_json(results):
    json_data = {'detected_instruments': results}
    return json.dumps(json_data, indent=2)

def export_pdf(results):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=16, style='B')
    pdf.cell(200, 10, txt="InstruNet AI - Analysis Report", ln=True, align='C')
    pdf.ln(10)
    pdf.set_font("Arial", size=12)
    for instrument, data in results.items():
        status = "Detected" if data['present'] else "Not Present"
        pdf.cell(200, 10, txt=f"{instrument}: {data['confidence']}% - {status}", ln=True)
    return pdf.output(dest='S').encode('latin-1')

# Main 3-column layout
col1, col2, col3 = st.columns([1.0, 2.2, 1.2])

# LEFT COLUMN - Upload Audio
with col1:
    st.markdown("### Upload Audio")
    
    uploaded_file = st.file_uploader(
        "Choose File",
        type=['wav', 'mp3', 'flac'],
        label_visibility="visible"
    )
    
    st.markdown("<small>wav, mp3, flac</small>", unsafe_allow_html=True)
    
    if uploaded_file:
        st.session_state.audio_file = uploaded_file
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Now Playing**")
        
        # Show waveform if analyzed
        if st.session_state.analyzed and st.session_state.analysis_results:
            fig_wave = create_waveform(
                st.session_state.analysis_results['y'],
                st.session_state.analysis_results['sr']
            )
            st.pyplot(fig_wave, use_container_width=True)
            plt.close()
        
        st.markdown(f"<small style='color: #2196F3;'>{uploaded_file.name}</small>", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("ANALYZE TRACK", use_container_width=True):
            with st.spinner("Analyzing audio..."):
                y, sr, instruments_detected = analyze_audio(uploaded_file)
                st.session_state.analyzed = True
                st.session_state.analysis_results = {
                    'y': y,
                    'sr': sr,
                    'instruments': instruments_detected
                }
                st.rerun()

# MIDDLE COLUMN - Analysis Results
with col2:
    st.markdown("### Analysis Results")
    
    if st.session_state.analyzed and st.session_state.analysis_results:
        results = st.session_state.analysis_results
        
        # Spectrogram
        fig_spec = create_spectrogram(results['y'], results['sr'])
        st.pyplot(fig_spec, use_container_width=True)
        plt.close()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Confidence bars
        fig_bars = create_confidence_bars(results['instruments'])
        st.plotly_chart(fig_bars, use_container_width=True)
    else:
        st.info("📁 Upload an audio file and click 'ANALYZE TRACK' to see results")

# RIGHT COLUMN - Detected Instruments
with col3:
    st.markdown("### Detected Instruments")
    
    if st.session_state.analyzed and st.session_state.analysis_results:
        instruments = st.session_state.analysis_results['instruments']
        
        # Show checkboxes
        for instrument, data in instruments.items():
            if data['present']:
                st.checkbox(instrument, value=True, disabled=True, key=f"check_{instrument}")
            else:
                col_check, col_label = st.columns([3, 2])
                with col_check:
                    st.checkbox(instrument, value=False, disabled=True, key=f"check_{instrument}")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Timeline
        fig_timeline = create_timeline(instruments)
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Export buttons
        col_json, col_pdf = st.columns(2)
        
        with col_json:
            json_data = export_json(instruments)
            st.download_button(
                label="EXPORT REPORT (JSON)",
                data=json_data,
                file_name="analysis_report.json",
                mime="application/json",
                use_container_width=True
            )
        
        with col_pdf:
            pdf_data = export_pdf(instruments)
            st.download_button(
                label="(PDF)",
                data=pdf_data,
                file_name="analysis_report.pdf",
                mime="application/pdf",
                use_container_width=True
            )
    else:
        st.info("Results will appear here after analysis")
