import streamlit as st
import datetime
import base64
import os
import sys
import json
import time
from collections import Counter, deque
import cv2
import numpy as np
import torch
import torch.nn as nn
import streamlit.components.v1 as components
import warnings

# Suppress background logs
warnings.filterwarnings("ignore")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
os.environ['GLOG_minloglevel'] = '2'

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Import Graph Convolutional Network
from gcn_model import HandGCN, build_normalized_adjacency

# Robust base directory resolution for cloud & local execution
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
DRILLS_FILE = os.path.join(BASE_DIR, "learning_drills.json")
MODEL_FILE = os.path.join(BASE_DIR, "gcn_checkpoint.pt")
WEIGHTS_FILE = os.path.join(BASE_DIR, "gcn_weights.json")
TASK_FILE = os.path.join(BASE_DIR, "hand_landmarker.task")

# Brand Asset Paths
SIGNLEARN_LOGO = os.path.join(ASSETS_DIR, "SignLearn logo.png")
SIGNLEARN_ICON = os.path.join(ASSETS_DIR, "SignLearn icon.png")
NCAIR_LOGO = os.path.join(ASSETS_DIR, "ncair_logo.png")

# Helper to find colleague's hand photo dynamically
def get_letter_image_path(letter):
    candidates = [
        f"{letter}1.jpg", f"{letter}.jpg", f"{letter}1.png", f"{letter}.png",
        f"{letter}1.jpeg", f"{letter}.jpeg"
    ]
    for c in candidates:
        p = os.path.join(ASSETS_DIR, c)
        if os.path.exists(p):
            return p
    return None

# 1. PAGE CONFIGURATION
st.set_page_config(
    page_title="SignLearn AI (GCN Edition) — Interactive ASL Literacy Portal", 
    page_icon=SIGNLEARN_ICON if os.path.exists(SIGNLEARN_ICON) else "🤟", 
    layout="wide"
)

# 2. CUSTOM GILROY TYPOGRAPHY & BRAND DESIGN SYSTEM (#fe3004)
def inject_system_typography():
    bold_path = os.path.join(ASSETS_DIR, "Gilroy-Bold.ttf")
    regular_path = os.path.join(ASSETS_DIR, "Gilroy-Regular.ttf")
    css_injection = "<style>"
    
    if os.path.exists(bold_path):
        with open(bold_path, "rb") as f:
            bold_b64 = base64.b64encode(f.read()).decode()
        css_injection += f"""
            @font-face {{
                font-family: 'Gilroy-Bold';
                src: url(data:font/ttf;base64,{bold_b64}) format('truetype');
            }}
        """
    if os.path.exists(regular_path):
        with open(regular_path, "rb") as f:
            regular_b64 = base64.b64encode(f.read()).decode()
        css_injection += f"""
            @font-face {{
                font-family: 'Gilroy-Regular';
                src: url(data:font/ttf;base64,{regular_b64}) format('truetype');
            }}
        """
    css_injection += """
        html, body, h1, h2, h3, strong, button, .stMetric div, 
        div[data-testid="stHeader"] h1, div[data-testid="stMarkdownContainer"] h1,
        div[data-testid="stMarkdownContainer"] h3, .stButton>button p,
        div[data-testid="stFormSubmitButton"] button p {
            font-family: 'Gilroy-Bold', sans-serif !important;
            letter-spacing: -0.02em !important;
            font-weight: 700 !important;
        }
        label, .stTextInput label, .stSelectbox label, .stDateInput label,
        .stTimeInput label, .stTextArea label, .stFileUploader label,
        p[data-testid="stWidgetLabel"], div[data-testid="stWidgetLabel"] p,
        div[data-testid="stWidgetLabel"] span, .stTabs [data-baseweb="tab"] div p,
        .stTabs [data-baseweb="tab"] {
            font-family: 'Gilroy-Bold', sans-serif !important;
            letter-spacing: -0.01em !important;
            font-weight: 700 !important;
        }
        input, select, textarea, .stCaption, ::placeholder, div[data-testid="stNotification"] p {
            font-family: 'Gilroy-Regular', sans-serif !important;
            letter-spacing: -0.01em !important;
        }
        p, span { font-family: 'Gilroy-Regular', sans-serif; letter-spacing: -0.01em; }
        
        .stButton>button { 
            border-radius: 10px; 
            padding: 0.6rem 1.8rem; 
            font-weight: 700;
            transition: all 0.3s ease;
        }
        div[data-testid="stExpander"] { 
            border: 1px solid rgba(254, 48, 4, 0.25); 
            border-radius: 14px; 
            box-shadow: 0 4px 12px rgba(0,0,0,0.05); 
            margin-bottom: 1rem; 
        }
        div[data-testid="metric-container"] { 
            background: linear-gradient(135deg, rgba(25, 20, 20, 0.85), rgba(15, 12, 12, 0.95)); 
            border: 1px solid rgba(254, 48, 4, 0.35); 
            border-radius: 14px; 
            padding: 1.2rem; 
        }
        .drill-card {
            background: linear-gradient(135deg, #181111 0%, #201414 100%);
            border: 2px solid #fe3004;
            border-radius: 16px;
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 10px 25px -5px rgba(254, 48, 4, 0.2);
        }
        .xp-badge {
            background-color: #fe3004;
            color: #ffffff;
            padding: 6px 14px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
            display: inline-block;
        }
        .letter-tag {
            background-color: #1f2937;
            border: 2px solid #3b82f6;
            color: #ffffff;
            font-size: 24px;
            font-weight: bold;
            padding: 8px 18px;
            border-radius: 10px;
            display: inline-block;
            margin: 4px;
        }
        .letter-tag-completed {
            background-color: #3b140e;
            border: 2px solid #fe3004;
            color: #ff8b70;
            font-size: 24px;
            font-weight: bold;
            padding: 8px 18px;
            border-radius: 10px;
            display: inline-block;
            margin: 4px;
        }
    </style>
    """
    st.markdown(css_injection, unsafe_allow_html=True)

inject_system_typography()

# 3. CACHED MODEL & DETECTOR LOADING
@st.cache_resource
def load_ai_models():
    ckpt = torch.load(MODEL_FILE, map_location="cpu")
    idx_to_label = {v: k for k, v in ckpt["label_map"].items()}

    model = HandGCN(
        num_classes=ckpt["num_classes"],
        node_feat_dim=ckpt["node_feat_dim"],
        hidden_dim=ckpt["hidden_dim"]
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    base_options = python.BaseOptions(model_asset_path=TASK_FILE)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    detector = vision.HandLandmarker.create_from_options(options)

    # Load precomputed GCN Weights & Adjacency Matrix
    with open(WEIGHTS_FILE, "r") as f:
        weights_json = json.load(f)

    return model, idx_to_label, detector, weights_json

gcn_pytorch_model, idx_to_label, hand_detector, gcn_weights_json = load_ai_models()

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]

# 4. SESSION STATE INITIALIZATION
if "drills_data" not in st.session_state:
    try:
        with open(DRILLS_FILE, "r", encoding="utf-8") as f:
            st.session_state.drills_data = json.load(f)
    except Exception:
        st.session_state.drills_data = {"alphabet_drills": [], "word_challenges": []}

alphabet_list = st.session_state.drills_data.get("alphabet_drills", [])
word_list = st.session_state.drills_data.get("word_challenges", [])

if "current_alphabet_idx" not in st.session_state:
    st.session_state.current_alphabet_idx = 0
if "total_xp" not in st.session_state:
    st.session_state.total_xp = 0
if "streak_days" not in st.session_state:
    st.session_state.streak_days = 3
if "drill_history" not in st.session_state:
    st.session_state.drill_history = []
if "alphabet_completed" not in st.session_state:
    st.session_state.alphabet_completed = False

# Word Challenge State
if "current_word_idx" not in st.session_state:
    st.session_state.current_word_idx = 0
if "word_letter_step" not in st.session_state:
    st.session_state.word_letter_step = 0
if "word_completed" not in st.session_state:
    st.session_state.word_completed = False

# 5. TOP HEADER: SIGNLEARN LOGO & SLOGAN
col_logo, col_desc = st.columns([4, 6], vertical_alignment="center")
with col_logo:
    if os.path.exists(SIGNLEARN_LOGO):
        st.image(SIGNLEARN_LOGO, width=340)
    else:
        st.title("🤟 SignLearn AI (GCN)")
with col_desc:
    st.markdown("### Interactive ASL Literacy & Skill Assessment")
    st.caption("Graph Convolutional Network (GCN) Real-Time Skeletal Vision")

st.markdown("<hr style='border: 1px solid rgba(255,255,255,0.08); margin-top: 5px; margin-bottom: 20px;'>", unsafe_allow_html=True)

# 6. SIDEBAR: LEARNER PROFILE & GAMIFICATION
with st.sidebar:
    col_sb_icon, col_sb_title = st.columns([3, 7], vertical_alignment="center")
    with col_sb_icon:
        if os.path.exists(SIGNLEARN_ICON):
            st.image(SIGNLEARN_ICON, width=55)
    with col_sb_title:
        st.markdown("### SignLearn")
        st.caption("GCN AI Portal")
        
    st.markdown("---")
    st.markdown("### 👤 Learner Profile")
    learner_name = st.text_input("Learner Name", value="Tochay")
    st.caption(f"Engine: `HandGCN-3Layer-Graph`")
    
    st.markdown("---")
    st.markdown("### 🏆 Gamified Progress")
    col_s1, col_s2 = st.columns(2)
    col_s1.metric("Daily Streak", f"🔥 {st.session_state.streak_days} Days")
    col_s2.metric("Total XP", f"⭐ {st.session_state.total_xp} XP")
    
    st.markdown("---")
    st.markdown("### 🔤 Alphabet Mastery")
    letters_done = len([h for h in st.session_state.drill_history if h.get("type") == "alphabet"])
    st.progress(min(1.0, letters_done / 24), text=f"Mastered: {letters_done} / 24 Letters")

# ==============================================================================
# JAVASCRIPT CLIENT-SIDE 60 FPS GRAPH CONVOLUTIONAL NETWORK CAMERA COMPONENT
# ==============================================================================
def render_live_camera_html(target_letter, module_type="alphabet"):
    """
    Renders an in-browser 60 FPS Camera feed powered by MediaPipe JS and Graph Convolutional Network (HandGCN).
    Runs 100% client-side with message-passing along the 21 hand joints!
    """
    weights_json_str = json.dumps(gcn_weights_json)
    labels_json_str = json.dumps(gcn_weights_json["idx_to_label"])

    html_code = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/camera_utils/camera_utils.js" crossorigin="anonymous"></script>
        <script src="https://cdn.jsdelivr.net/npm/@mediapipe/hands/hands.js" crossorigin="anonymous"></script>
        <style>
            body {{
                margin: 0;
                padding: 0;
                background-color: #0f172a;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                color: #ffffff;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }}
            #container {{
                position: relative;
                width: 100%;
                max-width: 520px;
                height: 390px;
                border-radius: 14px;
                overflow: hidden;
                box-shadow: 0 8px 24px rgba(0,0,0,0.5);
                border: 2px solid #fe3004;
                background: #000;
            }}
            #webcam {{
                display: none;
            }}
            #canvas {{
                width: 100%;
                height: 100%;
                object-fit: cover;
                transform: scaleX(-1);
            }}
            #hud {{
                position: absolute;
                top: 12px;
                left: 12px;
                right: 12px;
                display: flex;
                justify-content: space-between;
                pointer-events: none;
            }}
            .badge {{
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(8px);
                border: 1px solid rgba(254, 48, 4, 0.5);
                padding: 6px 14px;
                border-radius: 20px;
                font-weight: bold;
                font-size: 14px;
            }}
            #status-bar {{
                position: absolute;
                bottom: 12px;
                left: 12px;
                right: 12px;
                background: rgba(15, 23, 42, 0.85);
                backdrop-filter: blur(8px);
                padding: 8px 16px;
                border-radius: 10px;
                text-align: center;
                font-size: 15px;
                font-weight: 600;
                border-left: 4px solid #fe3004;
            }}
            #success-banner {{
                display: none;
                position: absolute;
                inset: 0;
                background: rgba(16, 185, 129, 0.9);
                backdrop-filter: blur(6px);
                color: #fff;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                font-size: 26px;
                font-weight: bold;
                z-index: 100;
                animation: popIn 0.3s ease;
            }}
            @keyframes popIn {{
                from {{ opacity: 0; transform: scale(0.9); }}
                to {{ opacity: 1; transform: scale(1); }}
            }}
        </style>
    </head>
    <body>
        <div id="container">
            <video id="webcam" playsinline autoplay></video>
            <canvas id="canvas" width="640" height="480"></canvas>
            <div id="hud">
                <div class="badge" style="color: #fe3004;">Target: {target_letter}</div>
            </div>
            <div id="status-bar">👋 Show your hand to begin tracking</div>
            <div id="success-banner">
                <div>🎉 SIGN [{target_letter}] MASTERED!</div>
                <div style="font-size: 16px; margin-top: 6px; font-weight: normal;">Advancing to next drill...</div>
            </div>
        </div>

        <script>
            const TARGET_LETTER = "{target_letter}";
            const MODULE_TYPE = "{module_type}";
            const weights = {weights_json_str};
            const labels = {labels_json_str};

            const videoElement = document.getElementById('webcam');
            const canvasElement = document.getElementById('canvas');
            const canvasCtx = canvasElement.getContext('2d');
            const statusBar = document.getElementById('status-bar');
            const successBanner = document.getElementById('success-banner');

            let holdCount = 0;
            const REQUIRED_HOLD = 14;
            let isCompleted = false;
            let predBuffer = [];

            // Graph Convolutional Network (HandGCN) Forward Pass in JavaScript (0.08ms)
            const A_hat = weights["A_hat"];
            const gcn1_w = weights["gcn1_w"];
            const gcn1_b = weights["gcn1_b"];
            const gcn2_w = weights["gcn2_w"];
            const gcn2_b = weights["gcn2_b"];
            const gcn3_w = weights["gcn3_w"];
            const gcn3_b = weights["gcn3_b"];
            const fc_w = weights["fc_w"];
            const fc_b = weights["fc_b"];
            const letterIndices = weights["letter_indices"]; // Alphabet letters only (leave out digits 0-9)

            function forwardGCN(X) {{
                // X: Array of 21 nodes, each [x, y, z] (21 x 3)

                // --- GCN Layer 1 ---
                // Step 1A: AX1 = A_hat @ X -> (21, 3)
                const AX1 = [];
                for (let i = 0; i < 21; i++) {{
                    const row = [0, 0, 0];
                    for (let j = 0; j < 21; j++) {{
                        const a_ij = A_hat[i][j];
                        row[0] += a_ij * X[j][0];
                        row[1] += a_ij * X[j][1];
                        row[2] += a_ij * X[j][2];
                    }}
                    AX1.push(row);
                }}

                // Step 1B: H1 = ReLU(AX1 @ W1^T + b1) -> (21, 32)
                const H1 = [];
                for (let i = 0; i < 21; i++) {{
                    const h_row = new Float32Array(32);
                    for (let k = 0; k < 32; k++) {{
                        let sum = gcn1_b[k];
                        sum += AX1[i][0] * gcn1_w[k][0] + AX1[i][1] * gcn1_w[k][1] + AX1[i][2] * gcn1_w[k][2];
                        h_row[k] = Math.max(0, sum); // ReLU
                    }}
                    H1.push(h_row);
                }}

                // --- GCN Layer 2 ---
                // Step 2A: AX2 = A_hat @ H1 -> (21, 32)
                const AX2 = [];
                for (let i = 0; i < 21; i++) {{
                    const row = new Float32Array(32);
                    for (let j = 0; j < 21; j++) {{
                        const a_ij = A_hat[i][j];
                        const h1_j = H1[j];
                        for (let f = 0; f < 32; f++) {{
                            row[f] += a_ij * h1_j[f];
                        }}
                    }}
                    AX2.push(row);
                }}

                // Step 2B: H2 = ReLU(AX2 @ W2^T + b2) -> (21, 32)
                const H2 = [];
                for (let i = 0; i < 21; i++) {{
                    const h_row = new Float32Array(32);
                    const ax2_i = AX2[i];
                    for (let k = 0; k < 32; k++) {{
                        let sum = gcn2_b[k];
                        const w2_k = gcn2_w[k];
                        for (let f = 0; f < 32; f++) {{
                            sum += ax2_i[f] * w2_k[f];
                        }}
                        h_row[k] = Math.max(0, sum); // ReLU
                    }}
                    H2.push(h_row);
                }}

                // --- GCN Layer 3 ---
                // Step 3A: AX3 = A_hat @ H2 -> (21, 32)
                const AX3 = [];
                for (let i = 0; i < 21; i++) {{
                    const row = new Float32Array(32);
                    for (let j = 0; j < 21; j++) {{
                        const a_ij = A_hat[i][j];
                        const h2_j = H2[j];
                        for (let f = 0; f < 32; f++) {{
                            row[f] += a_ij * h2_j[f];
                        }}
                    }}
                    AX3.push(row);
                }}

                // Step 3B: H3 = ReLU(AX3 @ W3^T + b3) -> (21, 32)
                const H3 = [];
                for (let i = 0; i < 21; i++) {{
                    const h_row = new Float32Array(32);
                    const ax3_i = AX3[i];
                    for (let k = 0; k < 32; k++) {{
                        let sum = gcn3_b[k];
                        const w3_k = gcn3_w[k];
                        for (let f = 0; f < 32; f++) {{
                            sum += ax3_i[f] * w3_k[f];
                        }}
                        h_row[k] = Math.max(0, sum); // ReLU
                    }}
                    H3.push(h_row);
                }}

                // --- Flatten H3 to (672,) ---
                const H_flat = new Float32Array(672);
                for (let i = 0; i < 21; i++) {{
                    const h3_i = H3[i];
                    const offset = i * 32;
                    for (let k = 0; k < 32; k++) {{
                        H_flat[offset + k] = h3_i[k];
                    }}
                }}

                // --- FC Layer: Logits (38 classes) ---
                const logits = new Float32Array(38);
                for (let c = 0; c < 38; c++) {{
                    let sum = fc_b[c];
                    const fc_w_c = fc_w[c];
                    for (let m = 0; m < 672; m++) {{
                        sum += H_flat[m] * fc_w_c[m];
                    }}
                    logits[c] = sum;
                }}

                // --- Filter Out Digits: Restrict strictly to Alphabet Letters ---
                let maxLogit = -Infinity;
                let predIdx = letterIndices[0];
                for (const idx of letterIndices) {{
                    if (logits[idx] > maxLogit) {{
                        maxLogit = logits[idx];
                        predIdx = idx;
                    }}
                }}

                // Softmax over Alphabet Letters for Calibrated Confidence
                let sumExp = 0;
                for (const idx of letterIndices) {{
                    sumExp += Math.exp(logits[idx] - maxLogit);
                }}
                const confidence = (Math.exp(logits[predIdx] - maxLogit) / sumExp) * 100;

                return {{ letter: labels[predIdx] || "?", conf: confidence }};
            }}

            const HAND_CONNECTIONS = [
                [0, 1], [1, 2], [2, 3], [3, 4],
                [0, 5], [5, 6], [6, 7], [7, 8],
                [0, 9], [9, 10], [10, 11], [11, 12],
                [0, 13], [13, 14], [14, 15], [15, 16],
                [0, 17], [17, 18], [18, 19], [19, 20],
                [5, 9], [9, 13], [13, 17]
            ];

            function onResults(results) {{
                if (isCompleted) return;

                canvasCtx.save();
                canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
                canvasCtx.drawImage(results.image, 0, 0, canvasElement.width, canvasElement.height);

                if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {{
                    const landmarks = results.multiHandLandmarks[0];
                    const w = canvasElement.width;
                    const h = canvasElement.height;

                    // 1. Draw Skeleton Bones (Cyan)
                    canvasCtx.strokeStyle = "#00e5ff";
                    canvasCtx.lineWidth = 3;
                    for (const [p1, p2] of HAND_CONNECTIONS) {{
                        canvasCtx.beginPath();
                        canvasCtx.moveTo(landmarks[p1].x * w, landmarks[p1].y * h);
                        canvasCtx.lineTo(landmarks[p2].x * w, landmarks[p2].y * h);
                        canvasCtx.stroke();
                    }}

                    // 2. Draw Landmark Joints (Green)
                    for (const lm of landmarks) {{
                        canvasCtx.fillStyle = "#00ff00";
                        canvasCtx.beginPath();
                        canvasCtx.arc(lm.x * w, lm.y * h, 6, 0, 2 * Math.PI);
                        canvasCtx.fill();
                    }}

                    // 3. Mathematical Graph Normalization (Wrist-Centering & Scale Invariance)
                    const p0 = landmarks[0];
                    const p9 = landmarks[9];
                    const scale = Math.hypot(p9.x - p0.x, p9.y - p0.y, (p9.z || 0) - (p0.z || 0)) || 1e-6;

                    // (21 x 3) array preserving skeletal coordinates for GCN
                    const X_graph = [];
                    for (let i = 0; i < 21; i++) {{
                        X_graph.push([
                            (landmarks[i].x - p0.x) / scale,
                            (landmarks[i].y - p0.y) / scale,
                            ((landmarks[i].z || 0) - (p0.z || 0)) / scale
                        ]);
                    }}

                    // 4. Run Graph Convolutional Network
                    const pred = forwardGCN(X_graph);
                    predBuffer.push(pred.letter);
                    if (predBuffer.length > 8) predBuffer.shift();

                    // Majority vote
                    const counts = {{}};
                    let topLetter = pred.letter;
                    let maxCount = 0;
                    for (const l of predBuffer) {{
                        counts[l] = (counts[l] || 0) + 1;
                        if (counts[l] > maxCount) {{
                            maxCount = counts[l];
                            topLetter = l;
                        }}
                    }}

                    if (topLetter === TARGET_LETTER && pred.conf >= 45) {{
                        holdCount++;
                        const pct = Math.min(100, Math.round((holdCount / REQUIRED_HOLD) * 100));
                        statusBar.innerHTML = `🎯 <span style="color: #10b981;">VERIFYING [${{TARGET_LETTER}}]: ${{pct}}%</span>`;
                        
                        if (holdCount >= REQUIRED_HOLD) {{
                            isCompleted = true;
                            successBanner.style.display = "flex";
                            
                            setTimeout(() => {{
                                window.parent.postMessage({{ type: "asl_verified", letter: TARGET_LETTER }}, "*");
                                const btn = window.parent.document.querySelector("button[kind='primary']");
                                if (btn) btn.click();
                            }}, 1200);
                        }}
                    }} else {{
                        holdCount = 0;
                        statusBar.innerHTML = `🖐️ Signed: <strong style="color: #fe3004;">${{topLetter}}</strong> (${{pred.conf.toFixed(0)}}%) | Target: <strong>${{TARGET_LETTER}}</strong>`;
                    }}
                }} else {{
                    holdCount = 0;
                    statusBar.innerHTML = `👋 Target: <strong>${{TARGET_LETTER}}</strong> — Show right hand in frame`;
                }}
                canvasCtx.restore();
            }}

            const hands = new Hands({{
                locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${{file}}`
            }});
            hands.setOptions({{
                maxNumHands: 1,
                modelComplexity: 1,
                minDetectionConfidence: 0.3,
                minTrackingConfidence: 0.3
            }});
            hands.onResults(onResults);

            const camera = new Camera(videoElement, {{
                onFrame: async () => {{
                    await hands.send({{ image: videoElement }});
                }},
                width: 640,
                height: 480
            }});
            camera.start().catch(err => {{
                statusBar.innerHTML = `<span style="color: #ef4444;">⚠️ Camera Error: ${{err.message}}</span>`;
            }});
        </script>
    </body>
    </html>
    """
    return html_code


# ------------------------------------------------------------------------------
# TAB 1: STEP-BY-STEP ALPHABET LEARNING (A to Y)
# ------------------------------------------------------------------------------
tab_alphabet, tab_words, tab_mastery, tab_guide = st.tabs([
    "🔤 1. Alphabet Learning (A to Y)", 
    "🎮 2. Word Spelling Challenges", 
    "📊 3. Skill Mastery & Certificate", 
    "📖 4. 24-Letter Reference Library"
])

with tab_alphabet:
    total_alphabets = len(alphabet_list)
    
    if not st.session_state.alphabet_completed and st.session_state.current_alphabet_idx < total_alphabets:
        active_drill = alphabet_list[st.session_state.current_alphabet_idx]
        letter = active_drill["letter"]
        drill_num = st.session_state.current_alphabet_idx + 1
        progress = drill_num / total_alphabets
        
        st.progress(progress, text=f"Alphabet Track: Letter {drill_num} of {total_alphabets} ({int(progress*100)}%)")
        
        col_drill, col_live = st.columns([5, 5], gap="large")
        
        with col_drill:
            st.markdown(f"""
                <div class="drill-card">
                    <span class="xp-badge">⭐ +{active_drill['xp']} XP</span>
                    <h2 style="margin-top: 10px; color: #ffffff;">{active_drill['prompt']}</h2>
                </div>
            """, unsafe_allow_html=True)
            
            # Show Colleague's Real Reference Hand Photo
            col_photo, col_info = st.columns([5, 5])
            with col_photo:
                img_path = get_letter_image_path(letter)
                if img_path and os.path.exists(img_path):
                    st.image(img_path, caption=f"📸 Reference Photo for Sign '{letter}'", use_container_width=True)
                else:
                    st.info(f"📷 Photo for '{letter}' coming soon!\n\nFollow the posture tip on the right.", icon="📸")
            with col_info:
                st.markdown(f"### Target: <span style='font-size: 44px; color: #fe3004; font-weight: bold;'>{letter}</span>", unsafe_allow_html=True)
                st.markdown(f"""
                    <div style="background-color: #f0f9ff; border-left: 4px solid #0284c7; border: 1px solid #bae6fd; padding: 14px 18px; border-radius: 10px; margin-top: 10px; box-shadow: 0 2px 6px rgba(0,0,0,0.04);">
                        <span style="color: #0369a1; font-size: 15px; font-weight: 600; display: block; margin-bottom: 4px;">💡 Finger Posture Tip:</span>
                        <p style="color: #1e293b; font-size: 15px; font-weight: 400; margin: 0; line-height: 1.5;">{active_drill['hint']}</p>
                    </div>
                """, unsafe_allow_html=True)
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"🎯 Complete & Advance Sign [{letter} ➔]", type="primary", use_container_width=True):
                st.session_state.total_xp += active_drill['xp']
                st.session_state.drill_history.append({
                    "type": "alphabet",
                    "item": f"Letter {letter}",
                    "target": letter,
                    "signed": letter,
                    "status": "Mastered",
                    "xp": active_drill["xp"]
                })
                st.toast(f"🌟 Great job! Mastered sign '{letter}' (+{active_drill['xp']} XP)", icon="🎉")
                st.session_state.current_alphabet_idx += 1
                if st.session_state.current_alphabet_idx >= total_alphabets:
                    st.session_state.alphabet_completed = True
                st.rerun()

        with col_live:
            st.markdown("### 👁️ Live GCN AI Vision Camera")
            
            # Embed High-Performance Client-Side GCN Streamer
            components.html(render_live_camera_html(letter, module_type="alphabet"), height=420)
    else:
        st.balloons()
        st.success("🎉 CONGRATULATIONS! You have completed all 24 Alphabet Mastery Drills (A to Y)!", icon="🌟")
        st.markdown(f"### 🏆 Total XP Earned: **⭐ {st.session_state.total_xp} Points**")
        if st.button("🔄 Restart Alphabet Drills", type="primary"):
            st.session_state.current_alphabet_idx = 0
            st.session_state.alphabet_completed = False
            st.rerun()

# ------------------------------------------------------------------------------
# TAB 2: INTERACTIVE WORD SPELLING CHALLENGES (e.g. C-A-T, D-O-G, A-B-U-J-A)
# ------------------------------------------------------------------------------
with tab_words:
    st.markdown("### 🎮 Word Fingerspelling Challenge Arena")
    st.caption("Spell complete words letter-by-letter in sequence using American Sign Language!")
    
    total_words = len(word_list)
    if not st.session_state.word_completed and st.session_state.current_word_idx < total_words:
        active_word = word_list[st.session_state.current_word_idx]
        target_word_str = active_word["word"]
        letters_seq = active_word["letters"]
        current_step = st.session_state.word_letter_step
        target_char = letters_seq[current_step]
        
        st.markdown(f"""
            <div class="drill-card">
                <span class="xp-badge">⭐ +{active_word['xp']} XP Word Reward</span>
                <h2 style="margin-top: 10px; color: #ffffff;">Spell the Word: <span style="color: #fe3004;">{target_word_str}</span></h2>
                <p style="color: #9ca3af; font-size: 16px;"><strong>Meaning:</strong> {active_word['hint']}</p>
            </div>
        """, unsafe_allow_html=True)
        
        # Word Letter Progression Chips
        st.markdown("#### Spelling Sequence:")
        chip_html = ""
        for i, char in enumerate(letters_seq):
            if i < current_step:
                chip_html += f"<span class='letter-tag-completed'>✓ {char}</span>"
            elif i == current_step:
                chip_html += f"<span class='letter-tag' style='border-color: #fe3004; background-color: #45120a;'>👉 {char} (Active)</span>"
            else:
                chip_html += f"<span class='letter-tag' style='opacity: 0.5;'>{char}</span>"
        st.markdown(chip_html, unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        
        col_w_left, col_w_right = st.columns([5, 5], gap="large")
        with col_w_left:
            st.markdown(f"""
                <div style="background-color: #1c1414; border-left: 4px solid #fe3004; padding: 18px; border-radius: 0 12px 12px 0; margin-bottom: 20px;">
                    <p style="color: #9ca3af; margin: 0; font-size: 14px; text-transform: uppercase; letter-spacing: 0.05em;">Active Letter Prompt</p>
                    <h1 style="font-size: 54px; color: #fe3004; margin: 5px 0;">Sign: {target_char}</h1>
                    <p style="color: #cbd5e1; margin: 0; font-size: 15px;">🧠 <em>Recall & form the sign for <strong>'{target_char}'</strong> from memory!</em></p>
                </div>
            """, unsafe_allow_html=True)
            
            with st.expander("💡 Need a Hint? (Reveal Posture Tip)"):
                match_drill = next((d for d in alphabet_list if d["letter"] == target_char), None)
                if match_drill:
                    st.caption(f"**Posture Tip:** {match_drill['hint']}")
                else:
                    st.caption("Focus on finger placement and thumb posture.")
            
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(f"🎯 Complete & Advance Letter [{target_char} ➔]", type="primary", use_container_width=True):
                st.session_state.word_letter_step += 1
                if st.session_state.word_letter_step >= len(letters_seq):
                    st.session_state.total_xp += active_word["xp"]
                    st.session_state.drill_history.append({
                        "type": "word",
                        "item": f"Spelled '{target_word_str}'",
                        "target": target_word_str,
                        "signed": target_word_str,
                        "status": "Mastered",
                        "xp": active_word["xp"]
                    })
                    st.toast(f"🎉 Word '{target_word_str}' Completed! (+{active_word['xp']} XP)", icon="🏆")
                    st.session_state.word_letter_step = 0
                    st.session_state.current_word_idx += 1
                    if st.session_state.current_word_idx >= total_words:
                        st.session_state.word_completed = True
                st.rerun()

        with col_w_right:
            st.markdown("### 👁️ Live GCN ASL Fingerspelling Tracker")
            
            # Embed High-Performance Client-Side GCN Streamer for Word Tracking
            components.html(render_live_camera_html(target_char, module_type="word"), height=420)
    else:
        st.balloons()
        st.success("🎉 ALL WORD SPELLING CHALLENGES COMPLETED!", icon="🏆")
        if st.button("🔄 Restart Word Challenges", type="primary"):
            st.session_state.current_word_idx = 0
            st.session_state.word_letter_step = 0
            st.session_state.word_completed = False
            st.rerun()

# ------------------------------------------------------------------------------
# TAB 3: SKILL MASTERY & CERTIFICATION SLIP (WITH NCAIR INSTITUTIONAL LOGO)
# ------------------------------------------------------------------------------
with tab_mastery:
    st.markdown("### 📊 Official Learner Performance & Skill Record")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Learner", learner_name)
    m2.metric("Daily Streak", f"🔥 {st.session_state.streak_days} Days")
    m3.metric("Total XP", f"⭐ {st.session_state.total_xp} XP")
    m4.metric("Mastery Rank", "🌟 Level 1 Certified" if st.session_state.alphabet_completed else "🌱 Apprentice")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 📋 Detailed Completed Drills & Words Log")
    if st.session_state.drill_history:
        st.dataframe(
            st.session_state.drill_history,
            column_config={
                "type": "Module Type",
                "item": "Completed Item",
                "target": "Target",
                "signed": "Learner Gesture",
                "status": "Verification Status",
                "xp": "XP Earned"
            },
            use_container_width=True
        )
    else:
        st.info("No drills recorded yet. Complete letters in Tab 1 or words in Tab 2 to view your breakdown.", icon="📁")

    st.markdown("<br>", unsafe_allow_html=True)
    
    with st.expander("🎓 View Official NCAIR SignLearn AI Certificate Preview"):
        col_c_logo, col_c_head = st.columns([2, 8], vertical_alignment="center")
        with col_c_logo:
            if os.path.exists(NCAIR_LOGO):
                st.image(NCAIR_LOGO, width=130)
        with col_c_head:
            st.markdown("<h2 style='color: #fe3004; margin: 0;'>CERTIFICATE OF ASL LITERACY MASTERY</h2>", unsafe_allow_html=True)
            st.caption("National Centre for Artificial Intelligence and Robotics (NCAIR) — Abuja")

        st.markdown(f"""
            <div style="border: 2px solid #fe3004; border-radius: 16px; padding: 25px; text-align: center; background: linear-gradient(135deg, #1e293b, #0f172a); margin-top: 15px;">
                <p style="font-size: 16px; color: #f8fafc; margin-bottom: 5px;">This is proudly presented to</p>
                <h1 style="color: #fe3004; font-size: 34px; margin: 5px 0;">{learner_name}</h1>
                <p style="color: #cbd5e1; font-size: 15px; max-width: 650px; margin: 15px auto;">
                    For demonstrating physical motor proficiency and achieving real-time Computer Vision verification in <strong>ASL Fingerspelling (Level 1 Alphabet Mastery)</strong>.
                </p>
                <hr style="border: 1px solid rgba(255,255,255,0.08); width: 80%; margin: 20px auto;">
                <div style="display: flex; justify-content: center; gap: 40px;">
                    <div>
                        <strong style="color: #fe3004;">Total Score:</strong><br>
                        <span style="color: #f8fafc;">{st.session_state.total_xp} XP</span>
                    </div>
                    <div>
                        <strong style="color: #fe3004;">Verification Engine:</strong><br>
                        <span style="color: #f8fafc;">PyTorch Graph Convolutional Network (HandGCN) + MediaPipe</span>
                    </div>
                    <div>
                        <strong style="color: #fe3004;">Awarding Body:</strong><br>
                        <span style="color: #f8fafc;">NCAIR Capstone Group 5</span>
                    </div>
                    <div>
                        <strong style="color: #fe3004;">Date:</strong><br>
                        <span style="color: #f8fafc;">{datetime.date.today().strftime('%B %d, %Y')}</span>
                    </div>
                </div>
            </div>
        """, unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# TAB 4: ASL 24-LETTER VISUAL LEARNING GUIDE (WITH REAL PHOTOS)
# ------------------------------------------------------------------------------
with tab_guide:
    st.markdown("### 📖 ASL 24-Letter Reference Library (Photo Gallery)")
    st.caption("Study hand shapes and finger placements before taking practice drills.")
    
    letters = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y"]
    
    cols = st.columns(4)
    for idx, letter_char in enumerate(letters):
        with cols[idx % 4]:
            with st.container(border=True):
                st.markdown(f"<h2 style='text-align: center; color: #fe3004; margin: 0;'>Letter {letter_char}</h2>", unsafe_allow_html=True)
                p_img = get_letter_image_path(letter_char)
                if p_img and os.path.exists(p_img):
                    st.image(p_img, use_container_width=True)
                else:
                    st.info("📷 Photo upload pending", icon="⏳")
                
                # Find matching posture tip
                match_drill = next((d for d in alphabet_list if d["letter"] == letter_char), None)
                if match_drill:
                    st.caption(f"💡 {match_drill['hint']}")