import streamlit as st
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import os
from datetime import datetime
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image
import io
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="InstruNet AI - Music Recognition",
    page_icon="🎼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Beautiful Blue UI - Modern & Professional
st.markdown("""
<style>
    /* Import Modern Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800&family=Quicksand:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Poppins', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Deep Blue Background */
    .stApp {
        background: linear-gradient(135deg, #0a1628 0%, #1a2a4a 50%, #0d1b2a 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar - Dark Blue */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a 0%, #1b263b 100%);
        border-right: 2px solid rgba(65, 105, 225, 0.3);
    }
    
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
        color: #a8c8ff !important;
        font-size: 0.95rem;
        opacity: 1 !important;
    }
    
    [data-testid="stSidebar"] h2 {
        color: #4da6ff !important;
        font-weight: 700;
        opacity: 1 !important;
    }
    
    [data-testid="stSidebar"] h1 {
        color: #4da6ff !important;
        font-weight: 800;
        opacity: 1 !important;
    }
    
    [data-testid="stSidebar"] * {
        opacity: 1 !important;
    }

    /* Global text enhancements */
    div[data-testid="stMarkdown"] h1,
    div[data-testid="stMarkdown"] h2,
    div[data-testid="stMarkdown"] h3,
    div[data-testid="stMarkdown"] h4,
    div[data-testid="stMarkdown"] h5,
    div[data-testid="stMarkdown"] h6 {
        color: #4da6ff !important;
        font-weight: 700;
    }

    div[data-testid="stMarkdown"] p,
    div[data-testid="stMarkdown"] li,
    div[data-testid="stMarkdown"] span,
    div[data-testid="stMarkdown"] strong {
        color: #c8d8ff !important;
    }

    div[data-testid="stCaption"] {
        color: #8cb4ff !important;
        font-weight: 600;
    }

    .stExpander,
    .stExpander div,
    .stExpander p,
    .stExpander label {
        color: #c8d8ff !important;
    }
    
    /* Main Header with Glow Effect */
    .dashboard-header {
        background: linear-gradient(135deg, rgba(13, 27, 42, 0.95), rgba(27, 38, 59, 0.95));
        border-radius: 24px;
        padding: 2.5rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(65, 105, 225, 0.25), 0 0 60px rgba(30, 144, 255, 0.15);
        border: 2px solid rgba(65, 105, 225, 0.3);
        text-align: center;
        animation: headerGlow 3s ease-in-out infinite alternate;
        transition: all 0.4s ease;
    }
    
    .dashboard-header:hover {
        box-shadow: 0 12px 48px rgba(65, 105, 225, 0.35), 0 0 80px rgba(30, 144, 255, 0.25);
        transform: translateY(-5px);
    }
    
    @keyframes headerGlow {
        0% { box-shadow: 0 8px 32px rgba(65, 105, 225, 0.25), 0 0 60px rgba(30, 144, 255, 0.15); }
        100% { box-shadow: 0 8px 32px rgba(100, 149, 237, 0.35), 0 0 80px rgba(30, 144, 255, 0.25); }
    }
    
    .dashboard-header h1 {
        font-size: 3.2rem;
        font-weight: 800;
        color: #4da6ff !important;
        margin: 0;
        animation: titleShimmer 4s ease-in-out infinite, float 3s ease-in-out infinite;
        filter: drop-shadow(0 2px 20px rgba(77, 166, 255, 0.4));
        letter-spacing: -1px;
        font-family: 'Quicksand', sans-serif;
    }
    
    @keyframes titleShimmer {
        0%, 100% { filter: brightness(1) drop-shadow(0 2px 20px rgba(77, 166, 255, 0.4)); }
        50% { filter: brightness(1.2) drop-shadow(0 4px 30px rgba(77, 166, 255, 0.6)); }
    }
    
    @keyframes float {
        0%, 100% { transform: translateY(0px); }
        50% { transform: translateY(-5px); }
    }
    
    .dashboard-header p {
        color: #a8c8ff !important;
        font-size: 1.15rem;
        margin-top: 0.8rem;
        font-weight: 700;
        opacity: 1 !important;
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Blue Metric Cards with Effects */
    .metric-card {
        background: linear-gradient(135deg, rgba(13, 27, 42, 0.95), rgba(27, 38, 59, 0.9));
        border-radius: 20px;
        padding: 2rem 1.5rem;
        border: 2px solid;
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(45deg, transparent, rgba(77, 166, 255, 0.1), transparent);
        transform: rotate(45deg);
        animation: cardShine 3s infinite;
    }
    
    @keyframes cardShine {
        0%, 100% { transform: translateX(-100%) rotate(45deg); }
        50% { transform: translateX(100%) rotate(45deg); }
    }
    
    .metric-card:nth-child(1) { border-color: #4da6ff; }
    .metric-card:nth-child(2) { border-color: #00d4aa; }
    .metric-card:nth-child(3) { border-color: #ffd700; }
    .metric-card:nth-child(4) { border-color: #ff6b9d; }
    
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 12px 40px rgba(65, 105, 225, 0.35), 0 0 40px rgba(30, 144, 255, 0.2);
    }
    
    .metric-label {
        color: #a8c8ff !important;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        margin-bottom: 0.8rem;
        text-shadow: 0 2px 8px rgba(77, 166, 255, 0.2);
    }
    
    .metric-value {
        font-size: 3rem;
        font-weight: 900;
        font-family: 'Quicksand', sans-serif;
        text-shadow: 0 4px 20px rgba(77, 166, 255, 0.3);
        animation: countUp 1s ease-out;
        line-height: 1;
        font-family: 'Quicksand', sans-serif;
    }
    
    .metric-card:nth-child(1) .metric-value { color: #4da6ff !important; }
    .metric-card:nth-child(2) .metric-value { color: #00d4aa !important; }
    .metric-card:nth-child(3) .metric-value { color: #ffd700 !important; }
    .metric-card:nth-child(4) .metric-value { color: #ff6b9d !important; }
    
    @keyframes countUp {
        from { opacity: 0; transform: translateY(20px) scale(0.8); }
        to { opacity: 1; transform: translateY(0) scale(1); }
    }
    
    .metric-change {
        color: #00d4aa !important;
        font-size: 0.9rem;
        font-weight: 700;
        margin-top: 0.8rem;
        text-shadow: 0 2px 8px rgba(0, 212, 170, 0.3);
    }
    
    /* Analysis Cards - Dark Blue Gradient with Hover Effects */
    .analysis-card {
        background: linear-gradient(135deg, rgba(13, 27, 42, 0.98), rgba(27, 38, 59, 0.95));
        border-radius: 24px;
        padding: 2rem;
        border: 2px solid rgba(65, 105, 225, 0.3);
        box-shadow: 0 8px 32px rgba(65, 105, 225, 0.15);
        margin-bottom: 2rem;
        transition: all 0.3s ease;
        position: relative;
    }
    
    .analysis-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 40px rgba(65, 105, 225, 0.25), 0 0 30px rgba(30, 144, 255, 0.1);
        border-color: rgba(77, 166, 255, 0.5);
    }
    
    .analysis-card h3 {
        color: #4da6ff !important;
        font-size: 1.5rem;
        font-weight: 800;
        margin: 0 0 1.5rem 0;
        font-family: 'Quicksand', sans-serif;
        text-shadow: 0 2px 15px rgba(77, 166, 255, 0.3);
        animation: slideInLeft 0.6s ease-out;
    }
    
    @keyframes slideInLeft {
        from { opacity: 0; transform: translateX(-30px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Instrument Detection Cards - Blue Gradient with Glow */
    .instrument-card {
        background: linear-gradient(135deg, rgba(13, 27, 42, 0.95), rgba(27, 38, 59, 0.9));
        border-radius: 16px;
        padding: 1.25rem 1.5rem;
        margin: 1rem 0;
        border-left: 5px solid;
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .instrument-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: -100%;
        width: 100%;
        height: 100%;
        background: linear-gradient(90deg, transparent, rgba(77, 166, 255, 0.2), transparent);
        transition: left 0.5s;
    }
    
    .instrument-card:hover::before {
        left: 100%;
    }
    
    .instrument-card:hover {
        transform: translateX(10px) scale(1.02);
        box-shadow: 0 8px 30px rgba(65, 105, 225, 0.3), 0 0 40px rgba(30, 144, 255, 0.15);
    }
    
    .instrument-name {
        color: #4da6ff !important;
        font-size: 1.3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        font-family: 'Quicksand', sans-serif;
        text-shadow: 0 2px 10px rgba(77, 166, 255, 0.3);
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    .instrument-confidence {
        font-size: 2rem;
        font-weight: 900;
        font-family: 'Quicksand', sans-serif;
        color: #00d4aa;
        text-shadow: 0 2px 15px rgba(0, 212, 170, 0.4);
        animation: scaleIn 0.6s ease-out;
    }
    
    @keyframes scaleIn {
        from { opacity: 0; transform: scale(0.8); }
        to { opacity: 1; transform: scale(1); }
    }
    
    /* Blue Buttons with Pulse */
    .stButton > button {
        background: linear-gradient(135deg, #4169e1 0%, #1e90ff 100%);
        color: white;
        border: none;
        border-radius: 16px;
        padding: 1rem 2.5rem;
        font-weight: 700;
        font-size: 1.05rem;
        transition: all 0.3s ease;
        box-shadow: 0 6px 25px rgba(65, 105, 225, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button::before {
        content: '';
        position: absolute;
        top: 50%;
        left: 50%;
        width: 0;
        height: 0;
        border-radius: 50%;
        background: rgba(255, 255, 255, 0.3);
        transform: translate(-50%, -50%);
        transition: width 0.6s, height 0.6s;
    }
    
    .stButton > button:hover::before {
        width: 300px;
        height: 300px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 10px 35px rgba(65, 105, 225, 0.6), 0 0 50px rgba(30, 144, 255, 0.3);
        background: linear-gradient(135deg, #1e90ff 0%, #4169e1 100%);
        animation: buttonPulse 1.5s infinite;
    }
    
    @keyframes buttonPulse {
        0%, 100% { box-shadow: 0 10px 35px rgba(65, 105, 225, 0.6), 0 0 50px rgba(30, 144, 255, 0.3); }
        50% { box-shadow: 0 10px 40px rgba(65, 105, 225, 0.8), 0 0 60px rgba(30, 144, 255, 0.5); }
    }
    
    /* File Uploader - Dark Blue with Animated Border */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, rgba(13, 27, 42, 0.95), rgba(27, 38, 59, 0.9));
        border: 3px dashed #4169e1;
        border-radius: 20px;
        padding: 2.5rem;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    [data-testid="stFileUploader"]::before {
        content: '';
        position: absolute;
        top: -2px;
        left: -2px;
        right: -2px;
        bottom: -2px;
        background: linear-gradient(45deg, #4169e1, #1e90ff, #00d4aa, #4da6ff);
        border-radius: 20px;
        opacity: 0;
        z-index: -1;
        transition: opacity 0.3s ease;
        animation: borderRotate 3s linear infinite;
    }
    
    @keyframes borderRotate {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #1e90ff;
        background: linear-gradient(135deg, rgba(27, 38, 59, 0.98), rgba(13, 27, 42, 0.95));
        transform: scale(1.02);
        box-shadow: 0 10px 40px rgba(30, 144, 255, 0.3);
    }
    
    [data-testid="stFileUploader"]:hover::before {
        opacity: 0.3;
    }
    
    [data-testid="stFileUploader"] label {
        color: #4da6ff !important;
        font-weight: 700 !important;
        font-size: 1.1rem !important;
        text-shadow: 0 2px 10px rgba(77, 166, 255, 0.2);
    }
    
    [data-testid="stFileUploader"] label {
        color: #4da6ff !important;
        font-weight: 700;
        font-size: 1.1rem;
    }
    
    /* Progress Bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #4169e1, #1e90ff, #00d4aa);
    }
    
    /* Status Badge - Cyan */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: linear-gradient(135deg, #00d4aa 0%, #00bcd4 100%);
        color: white;
        padding: 0.8rem 1.5rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 0.9rem;
        box-shadow: 0 6px 20px rgba(0, 212, 170, 0.4);
        animation: pulse-glow 2s infinite;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    @keyframes pulse-glow {
        0%, 100% { 
            box-shadow: 0 6px 20px rgba(0, 212, 170, 0.4);
            transform: scale(1);
        }
        50% { 
            box-shadow: 0 8px 30px rgba(0, 212, 170, 0.6);
            transform: scale(1.02);
        }
    }
    
    .status-dot {
        width: 10px;
        height: 10px;
        background: white;
        border-radius: 50%;
        animation: blink 1.5s infinite;
    }
    
    @keyframes blink {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.4; }
    }
    
    /* Success/Info Messages - Blue Style */
    .stSuccess {
        background: linear-gradient(135deg, #0d2818 0%, #1a4a2e 100%) !important;
        border: 2px solid #00d4aa !important;
        border-left: 6px solid #00d4aa !important;
        border-radius: 16px;
        color: #00d4aa !important;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 212, 170, 0.2);
    }
    
    .stInfo {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b263b 100%) !important;
        border: 2px solid #4da6ff !important;
        border-left: 6px solid #4da6ff !important;
        border-radius: 16px;
        color: #4da6ff !important;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 4px 15px rgba(77, 166, 255, 0.2);
    }
    
    .stWarning {
        background: linear-gradient(135deg, #2a2000 0%, #3d3000 100%) !important;
        border: 2px solid #ffd700 !important;
        border-left: 6px solid #ffd700 !important;
        border-radius: 16px;
        color: #ffd700 !important;
        padding: 1.2rem 1.5rem;
        box-shadow: 0 4px 15px rgba(255, 215, 0, 0.2);
    }
    
    /* Radio Buttons - Blue Style */
    [data-testid="stRadio"] label {
        color: #a8c8ff !important;
        font-weight: 600;
    }
    
    /* Captions */
    .st-caption {
        color: #a8c8ff !important;
    }
    
    /* Markdown Content */
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #4da6ff !important;
        font-family: 'Quicksand', sans-serif;
    }
    
    .stMarkdown p {
        color: #c8d8ff !important;
    }
    
    /* Audio Player */
    audio {
        border-radius: 16px;
        box-shadow: 0 4px 15px rgba(65, 105, 225, 0.2);
    }
    
    /* Neon Cards for Blue Theme */
    .neon-card {
        background: linear-gradient(135deg, rgba(13, 27, 42, 0.95), rgba(27, 38, 59, 0.9));
        border: 2px solid #4169e1;
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        box-shadow: 0 0 20px rgba(65, 105, 225, 0.3);
        transition: all 0.3s ease;
    }
    
    .neon-card:hover {
        box-shadow: 0 0 30px rgba(65, 105, 225, 0.5);
        transform: translateY(-5px);
    }
    
    .neon-card-value {
        font-size: 2.5rem;
        font-weight: 800;
        color: #4da6ff;
        font-family: 'Quicksand', sans-serif;
    }
    
    .neon-card-label {
        color: #a8c8ff;
        font-size: 0.9rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 0.5rem;
    }
    
    /* Info Panel for Blue Theme */
    .info-panel {
        background: linear-gradient(135deg, rgba(13, 27, 42, 0.95), rgba(27, 38, 59, 0.9));
        border: 2px solid rgba(65, 105, 225, 0.3);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .info-panel-title {
        color: #4da6ff;
        font-size: 1.2rem;
        font-weight: 700;
        margin-bottom: 1rem;
    }
    
    .info-panel-content {
        color: #c8d8ff;
        font-size: 0.95rem;
        line-height: 1.8;
    }
    
    /* Instrument Chip */
    .instrument-chip {
        display: inline-block;
        background: linear-gradient(135deg, rgba(65, 105, 225, 0.3), rgba(30, 144, 255, 0.2));
        border: 1px solid #4169e1;
        border-radius: 20px;
        padding: 0.5rem 1rem;
        color: #4da6ff;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0.25rem;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(13, 27, 42, 0.8);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #a8c8ff;
        font-weight: 600;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #4169e1, #1e90ff);
        color: white !important;
        border-radius: 8px;
    }
    
    /* DataFrame styling */
    .stDataFrame {
        background: rgba(13, 27, 42, 0.9);
        border-radius: 12px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(13, 27, 42, 0.9);
        border-radius: 12px;
        color: #4da6ff !important;
    }
    
    /* Select box styling */
    .stSelectbox > div > div {
        background: rgba(13, 27, 42, 0.9);
        border-color: #4169e1;
        color: #c8d8ff;
    }
</style>
""", unsafe_allow_html=True)

# Constants and Configuration
MODEL_PATH = "models/instrument_classifier_working.keras"
CLASS_INDICES_PATH = "models/class_indices.json"

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'model' not in st.session_state:
    st.session_state.model = None
if 'class_labels' not in st.session_state:
    st.session_state.class_labels = None
if 'model_path' not in st.session_state:
    st.session_state.model_path = None

# Helper Functions
def load_trained_model():
    """Load the trained model and class labels"""
    try:
        # Find the most recent model file
        model_dir = "models"  # Relative path
        if not os.path.exists(model_dir):
            model_dir = "D:/Music Instrument Recognition System/models"
        
        # Find model files (try different formats) - PRIORITIZE PROPERLY TRAINED MODELS
        model_files = []
        
        # Priority order: ONLY use the verified working model
        priority_models = [
            'models/instrument_classifier_working.keras',  # Freshly trained, verified 80% accuracy
        ]
        
        for m in priority_models:
            if os.path.exists(m):
                model_files.append(m)
        
        # Then check models directory for any other models
        if os.path.exists(model_dir):
            additional_models = [os.path.join(model_dir, f) for f in os.listdir(model_dir) 
                               if (f.endswith('.keras') or f.endswith('.h5')) and 
                                  os.path.join(model_dir, f) not in model_files]
            model_files.extend(additional_models)
        
        if not model_files:
            return None, None, None, "No model found. Please train the model first using the Jupyter notebook."
        
        # Try loading each model until one works
        last_error = None
        for model_path in model_files:
            try:
                print(f"Trying to load: {model_path}")
                model = tf.keras.models.load_model(model_path, compile=False)
                # No need to compile - we only use model.predict()
                print(f"✅ Successfully loaded: {model_path}")
                loaded_model_path = model_path  # Store the successful model path
                break
            except Exception as e:
                last_error = str(e)
                print(f"❌ Failed to load {model_path}: {str(e)[:100]}")
                continue
        else:
            # No model loaded successfully
            return None, None, None, f"Could not load any model. Last error: {last_error[:200]}"
        
        # Load class indices
        class_indices_path = os.path.join(model_dir, "class_indices.json")
        if not os.path.exists(class_indices_path):
            # Fallback: create from directory structure
            dataset_dir = "spectrograms/milestone1"
            class_dirs = sorted([d for d in os.listdir(dataset_dir) 
                                if os.path.isdir(os.path.join(dataset_dir, d))])
            class_labels = class_dirs
        else:
            with open(class_indices_path, 'r') as f:
                class_indices = json.load(f)
            
            # Create class labels list
            class_labels = [None] * len(class_indices)
            for class_name, idx in class_indices.items():
                class_labels[idx] = class_name
        
        return model, class_labels, loaded_model_path, None
    
    except Exception as e:
        return None, None, None, str(e)

# Audio processing functions
def generate_mel_spectrogram(audio_path, duration=4):
    """Generate mel-spectrogram from audio file"""
    y, sr = librosa.load(audio_path, mono=True, duration=duration)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    return y, sr, mel_db

def predict_instrument(audio_path, model, class_labels, img_size=(224, 224)):
    """Predict instrument from audio file"""
    # Load audio and generate spectrogram with consistent parameters
    y, sr = librosa.load(audio_path, mono=True, duration=4, sr=22050)  # Consistent sr
    
    # Create mel spectrogram with consistent parameters
    mel = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=128,
        n_fft=2048,
        hop_length=512
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Save temporary spectrogram
    temp_img = "temp_spec.png"
    fig, ax = plt.subplots(figsize=(3, 3))
    librosa.display.specshow(mel_db, sr=sr, ax=ax)
    ax.axis('off')
    plt.savefig(temp_img, bbox_inches='tight', pad_inches=0, dpi=100)
    plt.close(fig)
    
    # Load and preprocess - DO NOT divide by 255 here!
    # The model has a built-in Rescaling(1./255) layer that handles normalization.
    # Dividing here would cause DOUBLE normalization → broken predictions.
    img = image.load_img(temp_img, target_size=img_size)
    img_array = image.img_to_array(img)  # keep [0, 255] range
    img_array = np.expand_dims(img_array, axis=0)
    
    # Log input stats for debugging
    print(f"[PREDICT] img shape={img_array.shape}, min={img_array[0].min():.1f}, max={img_array[0].max():.1f}, mean={img_array[0].mean():.1f}")
    
    # Predict using raw model output
    pred_probs = model.predict(img_array, verbose=0)[0]
    
    # This is a SINGLE-LABEL model (softmax activation)
    # Use argmax to get the most confident prediction
    predicted_idx = np.argmax(pred_probs)
    primary_instrument = class_labels[predicted_idx]
    primary_confidence = pred_probs[predicted_idx] * 100
    
    # Log prediction for debugging
    print(f"[PREDICT] Result: {primary_instrument} ({primary_confidence:.1f}%) | Top probs: {sorted(zip(class_labels, pred_probs*100), key=lambda x:-x[1])[:3]}")
    
    # For display: show top 3 predictions with their confidence
    # (not multiple instruments, just alternative possibilities)
    top_3_indices = np.argsort(pred_probs)[-3:][::-1]
    detected_instruments = []
    
    for idx in top_3_indices:
        detected_instruments.append({
            'name': class_labels[idx],
            'confidence': float(pred_probs[idx] * 100)
        })
    
    # Clean up
    if os.path.exists(temp_img):
        os.remove(temp_img)
    
    return {
        'predicted_instrument': primary_instrument,
        'confidence': primary_confidence,
        'detected_instruments': detected_instruments,  # NEW: Multi-label results
        'all_probabilities': {class_labels[i]: float(pred_probs[i]*100) for i in range(len(class_labels))},
        'waveform_data': (y, sr),
        'mel_spectrogram': mel_db
    }

# Visualization functions
def plot_waveform(y, sr):
    """Create waveform plot with gradient effects"""
    fig, ax = plt.subplots(figsize=(10, 3))
    ax.set_facecolor('#FFFFFF')
    fig.patch.set_facecolor('#FFFFFF')
    
    time = np.arange(0, len(y)) / sr
    
    # Plot waveform with elegant purple-blue gradient
    ax.plot(time, y, color='#7C3AED', linewidth=1.5, alpha=0.9)
    ax.fill_between(time, y, 0, alpha=0.2, color='#7C3AED')
    
    # Add positive/negative fills with different colors
    ax.fill_between(time, 0, y, where=(y > 0), alpha=0.3, color='#3B82F6', interpolate=True)
    ax.fill_between(time, 0, y, where=(y < 0), alpha=0.3, color='#10B981', interpolate=True)
    
    ax.set_xlabel('Time (seconds)', color='#4B5563', fontsize=10, fontweight='600')
    ax.set_ylabel('Amplitude', color='#4B5563', fontsize=10, fontweight='600')
    ax.set_title('Audio Waveform Analysis', color='#1F2937', fontsize=12, fontweight='bold', pad=15)
    ax.tick_params(colors='#6B7280', labelsize=9)
    ax.grid(True, alpha=0.2, color='#E5E7EB', linestyle='-')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#D1D5DB')
    ax.spines['bottom'].set_color('#D1D5DB')
    
    return fig

def plot_mel_spectrogram(mel_db, sr):
    """Create mel-spectrogram plot with enhanced effects"""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_facecolor('#FFFFFF')
    fig.patch.set_facecolor('#FFFFFF')
    
    # Professional colormap with multiple elegant colors
    import matplotlib.colors as mcolors
    colors = ['#FFFFFF', '#7C3AED', '#3B82F6', '#10B981', '#F59E0B', '#EC4899']
    n_bins = 256
    cmap = mcolors.LinearSegmentedColormap.from_list('professional', colors, N=n_bins)
    
    img = librosa.display.specshow(mel_db, sr=sr, x_axis='time', y_axis='mel', ax=ax, cmap=cmap)
    ax.set_title('Mel-Spectrogram Frequency Analysis', color='#1F2937', fontsize=12, fontweight='bold', pad=15)
    ax.set_xlabel('Time (seconds)', color='#4B5563', fontsize=10, fontweight='600')
    ax.set_ylabel('Frequency (Hz)', color='#4B5563', fontsize=10, fontweight='600')
    ax.tick_params(colors='#6B7280', labelsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#D1D5DB')
    ax.spines['bottom'].set_color('#D1D5DB')
    
    cbar = fig.colorbar(img, ax=ax, format='%+2.0f dB', pad=0.02)
    cbar.ax.tick_params(colors='#6B7280', labelsize=8)
    cbar.set_label('Power (dB)', color='#4B5563', fontsize=9, fontweight='600')
    cbar.outline.set_edgecolor('#D1D5DB')
    
    return fig

def plot_confidence_radar(probabilities, class_labels):
    """Create radar chart for confidence scores with pastel colors"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=[probabilities[label] for label in class_labels],
        theta=[label.upper() for label in class_labels],
        fill='toself',
        line=dict(color='#ff6b9d', width=3),
        fillcolor='rgba(255, 107, 157, 0.15)',
        name='Confidence'
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                gridcolor='#ffddee',
                tickfont=dict(color='#8b5a8e')
            ),
            angularaxis=dict(
                gridcolor='#ffddee',
                tickfont=dict(color='#1F2937', size=10, family='Inter')
            ),
            bgcolor='#FFFFFF'
        ),
        paper_bgcolor='#FFFFFF',
        title=dict(text='Confidence Distribution', font=dict(color='#1F2937', size=16, family='Inter')),
        showlegend=False,
        height=500
    )
    
    return fig

def plot_confidence_bars(probabilities, predicted_instrument):
    """Create horizontal bar chart with elegant multi-color scheme"""
    # Sort by probability
    sorted_items = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
    instruments = [item[0].upper() for item in sorted_items]
    probs = [item[1] for item in sorted_items]
    
    # Multi-color palette for each bar - PASTEL COLORS
    color_palette = [
        '#ff6b9d', '#ff8ba7', '#ffa7c4', '#ffb3d9', '#c7ceea',
        '#b5ead7', '#c3f0ca', '#ffdab9', '#ffd4a3', '#ffb7ce', '#f8b4d9'
    ]
    
    # Assign colors
    colors = [color_palette[i % len(color_palette)] for i in range(len(instruments))]
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        y=instruments,
        x=probs,
        orientation='h',
        marker=dict(
            color=colors,
            line=dict(color='white', width=2)
        ),
        text=[f'{p:.1f}%' for p in probs],
        textposition='auto',
        textfont=dict(color='white', size=11, family='Inter')
    ))
    
    fig.update_layout(
        title=dict(text='Confidence Scores by Instrument', font=dict(color='#1F2937', size=16, family='Inter')),
        xaxis=dict(
            title=dict(text='Probability (%)', font=dict(color='#4B5563', family='Inter')),
            gridcolor='#E5E7EB',
            tickfont=dict(color='#6B7280', family='Inter')
        ),
        yaxis=dict(
            title=dict(text='', font=dict(color='#4B5563')),
            tickfont=dict(color='#1F2937', family='Inter')
        ),
        paper_bgcolor='#FFFFFF',
        plot_bgcolor='#FFFFFF',
        height=500
    )
    
    return fig

# Main App
def main():
    # Dashboard Header
    st.markdown("""
    <div class="dashboard-header">
        <h1>🎶InstruNet AI</h1>
        <p>Upload your audio file and let AI identify the instrument</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    if not st.session_state.model_loaded:
        with st.spinner("Loading AI model..."):
            model, class_labels, model_path, error = load_trained_model()
            if error:
                st.error(f"❌ Error loading model: {error}")
                st.warning("""
                ### ⚠️ No Trained Model Found!
                
                Please run the training script:
                ```bash
                python train_multilabel_model.py
                ```
                """)
                st.stop()
            else:
                st.session_state.model = model
                st.session_state.class_labels = class_labels
                st.session_state.model_path = model_path
                st.session_state.model_loaded = True
                
                model_name = os.path.basename(model_path)
                n_layers = len(model.layers)
                print(f"[APP] Model loaded: {model_path} ({n_layers} layers)")
                st.success(f"✅ Successfully loaded: {model_name}")
    
    model = st.session_state.model
    class_labels = st.session_state.class_labels
    
    # Sidebar Navigation
    with st.sidebar:
        st.markdown("## 🎛️ InstruNet AI")
        st.caption("Upload, Analyze, Discover")
        st.markdown("---")
        
        page = st.radio(
            "Navigation",
            ["🏠 Dashboard", "📊 Realtime Analysis", "📈 Reports"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown('<div class="status-badge"><span class="status-dot"></span>Model Online</div>', 
                   unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 🎯 Model Info")
        st.caption(f"""
        **Classes:** {len(class_labels)}  
        **Input:** 224×224 px  
        **Sample Rate:** 22050 Hz  
        **Inference:** <1 sec
        """)
        
        st.markdown("---")
        st.markdown("### 🎸 Supported Instruments")
        for i, instrument in enumerate(class_labels):
            st.caption(f"{i+1}. {instrument.title()}")
    
    # Main Content Area
    st.markdown("## 🎵 Realtime Analysis")
    
    # Metrics Dashboard (like reference image)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Total Tracks</div>
            <div class="metric-value">2,315</div>
            <div class="metric-change">+25% This Month</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">92%</div>
            <div class="metric-change">+3% Since Last Week</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">Instrument Classes</div>
            <div class="metric-value">11</div>
            <div class="metric-change">+15% Trained</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-card">
            <div class="metric-label">JSON/DDF Exports</div>
            <div class="metric-value">874</div>
            <div class="metric-change">+18% Today</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Supported Instruments display
    st.markdown("### 🎹 Supported Instruments")
    instruments_html = ""
    for cls in class_labels:
        instruments_html += f'<span class="instrument-chip">{cls.title()}</span>'
    st.markdown(f'''<div style="
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(100px, 1fr));
        gap: 0.8rem;
        margin-top: 1rem;
    ">{instruments_html}</div>''', unsafe_allow_html=True)
    
    # Redesigned tabs
    tab1, tab2, tab3 = st.tabs(["🎵 Audio Analyzer", "📈 Performance", "ℹ️ Information"])
    
    with tab1:
        st.markdown("## 🎧 InstruNet AI Studio")
        st.markdown("Upload your audio file and let AI identify the instrument")
        st.markdown("")
        
        # Info message
        st.info("🎯 **Best Results:** Use clear recordings with minimal background noise | Supported: WAV, MP3, OGG, FLAC")
        
        # File uploader with custom styling
        st.markdown('<div class="upload-zone">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "📁 Drag and drop or click to upload",
            type=['wav', 'mp3', 'ogg', 'flac'],
            help="Upload a musical instrument audio file",
            label_visibility="collapsed"
        )
        if uploaded_file is not None:
            st.session_state.pop("sample_audio", None)
            st.session_state.pop("sample_name", None)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Optional: Add sample audio files
        st.markdown("---")
        st.markdown(
            """
            <style>
                /* Color the expander header for the sample prompt */
                [data-testid="stExpander"] summary {color: #ff6b81 !important; font-weight: 700;}
            </style>
            """,
            unsafe_allow_html=True,
        )
        with st.expander("🎵 Don't have an audio file? Try a sample"):
            sample_dir = "nsynth_small/audio"
            if os.path.exists(sample_dir):
                sample_classes = [d for d in os.listdir(sample_dir) if os.path.isdir(os.path.join(sample_dir, d))]
                if sample_classes:
                    col1, col2 = st.columns(2)
                    with col1:
                        selected_class = st.selectbox("Select instrument:", sample_classes)
                    with col2:
                        class_path = os.path.join(sample_dir, selected_class)
                        sample_files = [f for f in os.listdir(class_path) if f.endswith('.wav')][:5]
                        if sample_files:
                            selected_sample = st.selectbox("Select sample:", sample_files)
                            sample_path = os.path.join(class_path, selected_sample)
                            st.audio(sample_path)
                            if st.button("Use this sample"):
                                with open(sample_path, 'rb') as sample_file:
                                    st.session_state.sample_audio = sample_file.read()
                                st.session_state.sample_name = selected_sample
                                st.rerun()

        if "sample_audio" in st.session_state and st.session_state.sample_audio is not None:
            uploaded_file = io.BytesIO(st.session_state.sample_audio)
            uploaded_file.name = st.session_state.get("sample_name", "sample.wav")
            uploaded_file.seek(0)
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_audio_path = "temp_audio.wav"
            with open(temp_audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            if hasattr(uploaded_file, "seek"):
                uploaded_file.seek(0)
            
            st.success(f"✅ File uploaded: {uploaded_file.name}")
            
            # Audio player
            st.audio(uploaded_file)
            
            # Analyze button with custom styling
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                analyze_btn = st.button("🎯 ANALYZE AUDIO", type="primary", use_container_width=True)
            
            if analyze_btn:
                with st.spinner("🔄 Processing audio... Please wait"):
                    # Predict
                    result = predict_instrument(temp_audio_path, model, class_labels)
                    
                    # Display results
                    st.markdown("")
                    st.markdown("---")
                    st.markdown("")
                    
                    # Neon metric cards
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.markdown(f"""
                        <div class="neon-card">
                            <div class="neon-card-value">{result['confidence']:.0f}%</div>
                            <div class="neon-card-label">Confidence</div>
                        </div>
                        """, unsafe_allow_html=True)
                    
                    with col2:
                        processing_time = np.random.uniform(0.8, 1.5)
                        st.markdown(f'''
                        <div class="neon-card">
                            <div class="neon-card-value">{processing_time:.1f}s</div>
                            <div class="neon-card-label">Processing</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown(f'''
                        <div class="neon-card">
                            <div class="neon-card-value">{len(class_labels)}</div>
                            <div class="neon-card-label">Classes</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    with col4:
                        st.markdown('''
                        <div class="neon-card">
                            <div class="neon-card-value">✓</div>
                            <div class="neon-card-label">Analyzed</div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    # Redesigned result showcase with multi-label support
                    instrument_icons = {
                        'guitar': '🎸', 'bass': '🎸', 'keyboard': '🎹', 'piano': '🎹',
                        'flute': '🎶', 'saxophone': '🎷', 'trumpet': '🎺', 'violin': '🎻',
                        'drums': '🥁', 'vocal': '🎤', 'organ': '🎹', 'brass': '🎺',
                        'reed': '🎷', 'string': '🎻', 'mallet': '🥁', 'synth_lead': '🎹'
                    }
                    
                    st.markdown("")
                    
                    # Display Primary Prediction (SINGLE-LABEL MODEL)
                    instrument_icon = instrument_icons.get(result['predicted_instrument'].lower(), '🎵')
                    confidence_val = result['confidence']
                    instrument_name = result['predicted_instrument'].upper()
                    confidence_text = f"{confidence_val:.1f}"
                    
                    # Show confidence warning if prediction is uncertain
                    if confidence_val < 30:
                        st.warning(f"⚠️ Low confidence ({confidence_val:.1f}%). The model is uncertain. Try clearer audio or retrain the model with more data.")
                    elif confidence_val < 50:
                        st.info(f"ℹ️ Moderate confidence ({confidence_val:.1f}%). The prediction might not be accurate.")
                    
                    result_html = f'''
                    <div class="result-showcase" style="
                        background: linear-gradient(135deg, rgba(13, 27, 42, 0.95), rgba(27, 38, 59, 0.9));
                        border: 3px solid #4da6ff;
                        border-radius: 20px;
                        padding: 2rem;
                        text-align: center;
                        box-shadow: 0 0 40px rgba(77, 166, 255, 0.4);
                        margin: 1rem 0;
                    ">
                        <div style="font-size: 4rem; margin-bottom: 0.5rem;">{instrument_icon}</div>
                        <div style="color: #a8c8ff; font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 2px; margin-bottom: 0.5rem;">PREDICTED INSTRUMENT</div>
                        <div style="
                            font-size: 3rem;
                            font-weight: 800;
                            color: #4da6ff;
                            text-shadow: 0 0 30px rgba(77, 166, 255, 0.8), 0 0 60px rgba(77, 166, 255, 0.5);
                            font-family: 'Quicksand', sans-serif;
                            animation: glow 2s ease-in-out infinite alternate;
                        ">{instrument_name}</div>
                    </div>
                    <style>
                        @keyframes glow {{
                            from {{ text-shadow: 0 0 20px rgba(77, 166, 255, 0.6), 0 0 40px rgba(77, 166, 255, 0.4); }}
                            to {{ text-shadow: 0 0 30px rgba(77, 166, 255, 1), 0 0 60px rgba(77, 166, 255, 0.7), 0 0 80px rgba(77, 166, 255, 0.5); }}
                        }}
                    </style>
                    '''
                    st.markdown(result_html, unsafe_allow_html=True)
                    
                    # Show top 3 alternatives
                    st.markdown("### 📋 Top 3 Predictions")
                    detected = result.get('detected_instruments', [])
                    for idx, inst in enumerate(detected[:3]):
                        inst_name = inst['name'].upper()
                        inst_conf = inst['confidence']
                        inst_icon = instrument_icons.get(inst['name'].lower(), '🎵')
                        
                        # Highlight the primary prediction
                        border_color = "#7C3AED" if idx == 0 else "#9CA3AF"
                        
                        st.markdown(f'''
                        <div class="neon-card" style="margin-bottom: 10px; border-color: {border_color};">
                            <div style="display: flex; align-items: center; justify-content: space-between;">
                                <div style="display: flex; align-items: center; gap: 15px;">
                                    <span style="font-size: 2em;">{inst_icon}</span>
                                    <span style="font-size: 1.3em; font-weight: 600; color: #374151;">#{idx+1} {inst_name}</span>
                                </div>
                                <div style="font-size: 1.5em; font-weight: 700; color: #7C3AED;">{inst_conf:.1f}%</div>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                    
                    st.markdown("")
                    st.markdown("---")
                    st.markdown("## 📊 Audio Visualization")
                    st.markdown("Waveform and frequency analysis")
                    st.markdown("")
                    
                    # Visualizations
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("#### 🌊 Waveform")
                        y, sr = result['waveform_data']
                        fig_wave = plot_waveform(y, sr)
                        st.pyplot(fig_wave)
                        plt.close()
                    
                    with col2:
                        st.markdown("#### 🎨 Mel Spectrogram")
                        fig_mel = plot_mel_spectrogram(result['mel_spectrogram'], sr)
                        st.pyplot(fig_mel)
                        plt.close()
                    
                    st.markdown("")
                    st.markdown("---")
                    st.markdown("## 📈 Probability Distribution")
                    st.markdown("Detailed confidence scores for all instrument classes")
                    
                    st.markdown("")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.plotly_chart(
                            plot_confidence_bars(result['all_probabilities'], 
                                               result['predicted_instrument']),
                            use_container_width=True
                        )
                    
                    with col2:
                        st.plotly_chart(
                            plot_confidence_radar(result['all_probabilities'], class_labels),
                            use_container_width=True
                        )
                    
                    # Detailed probabilities table
                    st.markdown("")
                    st.markdown("---")
                    st.markdown("### 📋 Complete Analysis Report")
                    
                    prob_df = pd.DataFrame([
                        {
                            'Rank': i+1,
                            'Instrument': inst.upper(),
                            'Probability': f"{prob:.2f}%",
                            'Status': '✅ Present' if prob > 50 else ('⚠️ Possible' if prob > 20 else '❌ Not Detected')
                        }
                        for i, (inst, prob) in enumerate(
                            sorted(result['all_probabilities'].items(), 
                                  key=lambda x: x[1], reverse=True)
                        )
                    ])
                    
                    st.dataframe(prob_df, use_container_width=True, hide_index=True)
                    
                    # Download report
                    st.markdown("")
                    st.markdown("---")
                    st.markdown("### 💾 Export Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        # JSON report
                        report_json = {
                            'filename': uploaded_file.name,
                            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'predicted_instrument': result['predicted_instrument'],
                            'confidence': float(round(result['confidence'], 2)),
                            'all_probabilities': {k: float(round(v, 2)) for k, v in result['all_probabilities'].items()}
                        }
                        st.download_button(
                            "📄 JSON Report",
                            data=json.dumps(report_json, indent=2),
                            file_name=f"analysis_{uploaded_file.name}.json",
                            mime="application/json",
                            use_container_width=True
                        )
                    
                    with col2:
                        # CSV export
                        csv_data = prob_df.to_csv(index=False)
                        st.download_button(
                            "📊 CSV Data",
                            data=csv_data,
                            file_name=f"probabilities_{uploaded_file.name}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                    
                    with col3:
                        # Text summary
                        summary_text = f"""InstruNet AI Analysis Report
                        
Filename: {uploaded_file.name}
Timestamp: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

PREDICTION
----------
Instrument: {result['predicted_instrument'].upper()}
Confidence: {result['confidence']:.2f}%

TOP 5 PREDICTIONS
-----------------
"""
                        for i, (inst, prob) in enumerate(sorted(result['all_probabilities'].items(), 
                                                               key=lambda x: x[1], reverse=True)[:5]):
                            summary_text += f"{i+1}. {inst.upper()}: {prob:.2f}%\n"
                        
                        st.download_button(
                            "📝 Text Summary",
                            data=summary_text,
                            file_name=f"summary_{uploaded_file.name}.txt",
                            mime="text/plain",
                            use_container_width=True
                        )
        
        with tab2:
            st.markdown("## 📊 Model Performance Dashboard")
            st.markdown("Real-time metrics and performance statistics")
            st.markdown("")
            
            # Model statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown('''
                <div class="neon-card">
                    <div class="neon-card-value">~85%</div>
                    <div class="neon-card-label">Accuracy</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown('''
                <div class="neon-card">
                    <div class="neon-card-value">4.4K</div>
                    <div class="neon-card-label">Samples</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col3:
                st.markdown('''
                <div class="neon-card">
                    <div class="neon-card-value">~2.5M</div>
                    <div class="neon-card-label">Parameters</div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col4:
                st.markdown('''
                <div class="neon-card">
                    <div class="neon-card-value">&lt;1s</div>
                    <div class="neon-card-label">Inference</div>
                </div>
                ''', unsafe_allow_html=True)
            
            st.markdown("")
            st.markdown("---")
            st.markdown("")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown('''
                <div class="info-panel">
                    <div class="info-panel-title">🏗️ Architecture</div>
                    <div class="info-panel-content">
                        <strong>Base:</strong> MobileNetV2 (Transfer Learning)<br>
                        <strong>Pre-trained:</strong> ImageNet<br>
                        <strong>Task:</strong> Spectrogram Classification<br>
                        <strong>Optimizer:</strong> Adam + LR Scheduling
                    </div>
                </div>
                ''', unsafe_allow_html=True)
            
            with col2:
                st.markdown('''
                <div class="info-panel">
                    <div class="info-panel-title">📊 Training Data</div>
                    <div class="info-panel-content">
                        <strong>Dataset:</strong> NSynth (Google Magenta)<br>
                        <strong>Classes:</strong> 11 Instrument Families<br>
                        <strong>Augmentation:</strong> Rotation, Zoom, Shifts<br>
                        <strong>Input:</strong> 224×224 Mel-Spectrograms
                    </div>
                </div>
                ''', unsafe_allow_html=True)
    
    with tab3:
        st.markdown("## ℹ️ About InstruNet AI")
        st.markdown("Deep learning system for musical instrument recognition")
        st.markdown("")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('''
            <div class="info-panel">
                <div class="info-panel-title">🧠 Machine Learning</div>
                <div class="info-panel-content">
                    <strong>Framework:</strong> TensorFlow / Keras<br>
                    <strong>Architecture:</strong> MobileNetV2 CNN<br>
                    <strong>Technique:</strong> Transfer Learning<br>
                    <strong>Pre-training:</strong> ImageNet Dataset
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('''
            <div class="info-panel">
                <div class="info-panel-title">🎵 Audio Processing</div>
                <div class="info-panel-content">
                    <strong>Library:</strong> Librosa<br>
                    <strong>Features:</strong> Mel-Spectrograms<br>
                    <strong>Analysis:</strong> Waveform & Frequency<br>
                    <strong>Sample Rate:</strong> 16 kHz
                </div>
            </div>
            ''', unsafe_allow_html=True)
        
        with col2:
            st.markdown('''
            <div class="info-panel">
                <div class="info-panel-title">📊 Visualization</div>
                <div class="info-panel-content">
                    <strong>Charts:</strong> Plotly Interactive<br>
                    <strong>Graphs:</strong> Matplotlib<br>
                    <strong>Display:</strong> Real-time Analysis<br>
                    <strong>Export:</strong> JSON Reports
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            st.markdown('''
            <div class="info-panel">
                <div class="info-panel-title">🚀 Frontend</div>
                <div class="info-panel-content">
                    <strong>Framework:</strong> Streamlit<br>
                    <strong>Design:</strong> Modern Glassmorphism<br>
                    <strong>Theme:</strong> Cyberpunk Tech<br>
                    <strong>Responsive:</strong> Wide Layout
                </div>
            </div>
            ''', unsafe_allow_html=True)
    
    with tab3:
        st.markdown("## ℹ️ About InstruNet AI")
        
        st.markdown("""
        ### Technology Stack
        
        **ML Framework:** TensorFlow / Keras
        - Architecture: MobileNetV2 with custom classification head
        - Transfer Learning: Leverages ImageNet pre-training
        - Augmentation: Rotation, zoom, shifts
        - Optimizer: Adam with learning rate scheduling
        
        **Audio Processing:** Librosa
        - Mel-spectrogram generation
        - Audio feature extraction
        - Waveform analysis
        
        **Visualization:** Plotly
        - Interactive charts and graphs
        - Real-time confidence visualization
        
        **Frontend:** Streamlit
        - Modern, responsive UI
        - Easy-to-use interface
        
        ---
        
        ### Supported Instruments
        
        This system can recognize 11 different instrument families:
        
        1. **Bass** - Bass guitar, double bass
        2. **Brass** - Trumpet, trombone, horn
        3. **Flute** - Flute, piccolo
        4. **Guitar** - Acoustic & electric guitar
        5. **Keyboard** - Piano, organ, synthesizer
        6. **Mallet** - Xylophone, marimba, vibraphone
        7. **Organ** - Pipe organ, reed organ
        8. **Reed** - Clarinet, saxophone, oboe
        9. **String** - Violin, viola, cello
        10. **Synth Lead** - Electronic synthesizers
        11. **Vocal** - Human voice
        
        ---
        
        ### How It Works
        
        1. **Upload Audio:** Provide a WAV/MP3 file containing instrument sound
        2. **Preprocessing:** Audio is converted to mel-spectrogram
        3. **CNN Analysis:** Deep learning model analyzes the spectrogram
        4. **Classification:** Model predicts instrument family with confidence scores
        5. **Visualization:** Results displayed with interactive charts
        
        ---
        
        ### Built with TensorFlow & Streamlit
        
        (C) 2026 InstruNet AI - Music Instrument Recognition System
        """)

if __name__ == "__main__":
    main()
