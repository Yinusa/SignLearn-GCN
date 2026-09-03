<p align="center">
  <img src="assets/SignLearn logo.png" alt="SignLearn AI logo" width="360">
</p>

<h1 align="center">SignLearn AI — GCN Edition</h1>

<p align="center">
  A real-time, gamified American Sign Language (ASL) fingerspelling tutor,<br>
  powered by a Graph Convolutional Network over the hand's skeletal structure.<br>
  Built by NCAIR Capstone Group 5 at the National Centre for Artificial Intelligence and Robotics (NCAIR), Abuja, Nigeria.
</p>

---

## What it does

SignLearn AI watches a learner's hand through a webcam, recognizes which ASL letter they're fingerspelling, and confirms it only once the sign has been held correctly and steadily — like a live tutor that never gets tired of correcting you. Get it right, earn XP; build a streak; work through the full alphabet and word-spelling drills; unlock a skill-mastery certificate.

This version classifies hand shape using a **Graph Convolutional Network (GCN)** rather than a plain feed-forward network. Instead of treating the 21 MediaPipe hand landmarks as an unordered list of 63 numbers, the model represents the hand as a graph — 21 joints as nodes, connected the same way MediaPipe's own skeleton connects them — and lets each joint's representation absorb information from its anatomically connected neighbors before classifying the hand shape. The whole pipeline still runs **client-side, in the browser**: the trained weights are exported to JSON and the graph-convolution forward pass is re-implemented by hand in JavaScript, so there's zero server round-trip per frame.

## Features

- **Live camera recognition** — 24 static ASL letters (A–Y, excluding the motion letters J and Z) are drilled by the app, confirmed via a majority-vote smoothing + hold-to-confirm pipeline so a single noisy frame can't trigger a false match.
- **Alphabet drills** — work through all 24 letters one at a time, with a reference photo and posture hint alongside the live camera.
- **Word spelling challenges** — spell full words (e.g. CAT, DOG, ABUJA, BOLA, LOVE) letter by letter.
- **Gamification** — XP per drill, streak tracking, a full drill history log, and a mastery rank (Practicing Learner → Apprentice → Skilled → Master Signer) computed from session accuracy.
- **Certificate preview** — a branded "Certificate of ASL Literacy Mastery" summarizing a learner's progress.
- **Reference library** — a photo gallery of all 24 letters with posture tips, for study before attempting a drill.
- **Zero-lag, zero-backend inference** — the GCN's forward pass (graph propagation + linear layers) is re-implemented in plain JavaScript, so recognition runs entirely in the learner's browser, no GPU or server required.

## How it works

```
Webcam frame
    │
    ▼
MediaPipe Hand Landmarker  →  21 joints × (x, y, z)
    │
    ▼
Normalize (wrist-centered, scaled by a fixed bone length) — kept as a 21×3 array
    │
    ▼
HandGCN classifier:
    3 × [ propagate along hand skeleton (Â · x)  →  learned linear layer  →  ReLU ]
    → flatten (21 × 32 → 672)  →  final linear layer → 38 class logits
    │
    ▼
Filter to the 26 alphabet letters, take the top one + softmax confidence
    │
    ▼
Majority-vote smoothing over recent frames
    │
    ▼
Hold-to-confirm (14 consecutive frames matching the target, ≥45% confidence)  →  confirmed letter
    │
    ▼
Gamification layer (XP, streak, drill history, mastery rank)
```

The graph the model reasons over is the hand's own skeleton — thumb, index, middle, ring, and pinky chains, plus the palm arcs connecting each finger's base — with a fixed, precomputed adjacency matrix (`Â = D⁻¹ᐟ²(A+I)D⁻¹ᐟ²`, the standard Kipf & Welling GCN propagation rule) reused for every prediction, since every hand sample has the same 21 joints wired the same way.

> **Note:** the repo includes **two** implementations of the pipeline, but only one is currently working end-to-end:
>
> | | `main.py` (deployed) | `signlearn_engine.py` (standalone) |
> |---|---|---|
> | Runs | In the browser, via an embedded HTML/JS component | Locally, as a Python + OpenCV desktop script |
> | Model | `HandGCN` (this repo's actual model) | ⚠️ Still references the **old MLP** architecture and looks for `mlp_model_final.pth`, which doesn't exist in this repo |
> | Status | ✅ Working | ❌ Broken — needs to be updated to build `HandGCN` and load `gcn_checkpoint.pt` before it will run |

## Project structure

```
SignLearn-GCN/
├── main.py                 # Streamlit app — the deployed web tutor (uses the GCN)
├── signlearn_engine.py      # Standalone desktop OpenCV engine + SignLearnEngine (gamification) class
├── gcn_model.py              # HandGCN architecture definition (graph, adjacency matrix, GCN layers)
├── gcn_checkpoint.pt         # Trained GCN weights, PyTorch format
├── gcn_weights.json          # Trained GCN weights + adjacency matrix, exported for the browser-side JS forward pass
├── hand_landmarker.task      # MediaPipe hand landmark detection model
├── label_map.json            # App-level letter mapping used for drills/reference images (24 letters, A–Y excl. J/Z)
├── learning_drills.json      # Alphabet drills and word-spelling challenges (targets, hints, XP values)
├── assets/                   # Logo, icons, letter reference photos, Gilroy font files
├── requirements.txt           # Python dependencies
├── packages.txt                # System-level (apt) dependencies for cloud deployment
├── runtime.txt                  # Pinned Python version for deployment
└── .python-version               # Local Python version pin
```

## Getting started

### Prerequisites

- Python 3.11 (see `.python-version` / `runtime.txt`)
- A webcam
- On Linux, the system libraries listed in `packages.txt` — installed automatically on most cloud hosts, install manually on a bare Linux machine

### Installation

```bash
git clone "https://github.com/Yinusa/SignLearn-GCN"
cd SignLearn-GCN
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Running the app

```bash
streamlit run main.py
```

Streamlit will open the app in your browser. Allow camera access when prompted — recognition runs locally in your browser, so no video is sent to a server.

### Running the standalone desktop engine

Not currently functional — `signlearn_engine.py` still builds the old MLP architecture and looks for a `mlp_model_final.pth` checkpoint that isn't part of this repo. To fix it, update its model-loading code to construct `HandGCN` (from `gcn_model.py`) and load `gcn_checkpoint.pt` instead.

## Model details

- **Input:** 21 hand landmarks, each a normalized (x, y, z) triple — wrist-centered, scaled by the wrist-to-middle-finger-knuckle distance. Kept as a 21×3 structure (not flattened) so the graph layers can operate on it.
- **Architecture:** `HandGCN` — 3 stacked graph-convolution layers (propagate along the hand skeleton via a fixed normalized adjacency matrix, then a learned linear layer, then ReLU), hidden dimension **32**, followed by flattening all 21 joints' final representations (672 values) into a single linear classification layer. Dropout of 0.3 before the final layer.
- **Trained class space:** 38 classes — digits 0–9, all 26 letters A–Z (including J and Z), plus `del` and `space` control gestures.
- **What the app actually uses:** predictions are filtered down to just the 26 alphabet letters before being shown to a learner; drill content only ever targets the same 24 non-motion letters as before (J and Z are excluded at the curriculum level, since they're motion signs a static frame can't capture — even though the model technically has classes for them).

## Known limitations

- The standalone desktop engine (`signlearn_engine.py`) is currently broken — see the note above.
- J and Z are trained classes in the model but are not used in any drill.
- The desktop and web engines (once the desktop one is fixed) may use different tuning constants (smoothing window, hold-frame count, confidence threshold) — worth double-checking once `signlearn_engine.py` is updated.
- The "Complete & Advance" button in the web app can be clicked manually regardless of camera confirmation — there's no server-side enforcement that a drill was completed via the camera.
- The "ABUJA" word challenge omits the letter J, consistent with J being excluded from the app's drill content.

## Acknowledgments

Built as a capstone project at the [National Centre for Artificial Intelligence and Robotics (NCAIR)](https://ncair.nitda.gov.ng/), Abuja, Nigeria.
