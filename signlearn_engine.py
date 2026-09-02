import json
import os
import sys
import time
from datetime import datetime
from collections import Counter, deque
import cv2
import numpy as np
import torch
import torch.nn as nn
import warnings

# Suppress sklearn/C++ log warnings
warnings.filterwarnings("ignore")
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')
os.environ['GLOG_minloglevel'] = '2'

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class SignLearnEngine:
    """
    State Machine & Gamified Skill Scoring Engine for SignLearn AI (Duolingo for ASL).
    """
    def __init__(self, drills_file="learning_drills.json"):
        if not os.path.exists(drills_file):
            drills_file = os.path.join("asl_project", "learning_drills.json")

        
        self.drills_file = drills_file
        self.drills = self._load_drills()
        
        # Learner State & Gamification
        self.learner_name = "Tochay"
        self.learner_id = "SL-NCAIR-042"
        self.current_drill_index = 0
        self.total_xp = 0
        self.streak_days = 3
        self.drills_log = []
        self.start_time = None
        self.end_time = None

    def _load_drills(self):
        """Loads learning prompts from JSON."""
        if not os.path.exists(self.drills_file):
            raise FileNotFoundError(f"Drills file '{self.drills_file}' not found!")
        with open(self.drills_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def start_session(self, learner_name="Tochay", learner_id="SL-NCAIR-042"):
        """Initializes a new learning and practice session."""
        self.learner_name = learner_name
        self.learner_id = learner_id
        self.current_drill_index = 0
        self.total_xp = 0
        self.drills_log = []
        self.start_time = time.time()
        self.end_time = None
        print(f"\n[+] SignLearn Session Started for {self.learner_name} (ID: {self.learner_id})")
        print(f"[+] Total Drills: {len(self.drills)} | Daily Streak: 🔥 {self.streak_days} Days")

    def get_current_drill(self):
        """Returns the active learning drill dictionary."""
        if self.current_drill_index < len(self.drills):
            return self.drills[self.current_drill_index]
        return None

    def submit_sign(self, candidate_sign):
        """
        Receives an ASL sign from the camera, verifies accuracy against the target,
        awards XP points, and advances to the next drill.
        """
        if self.is_finished():
            return None

        current_drill = self.drills[self.current_drill_index]
        target = current_drill["target_letter"].strip().upper()
        user_sign = candidate_sign.strip().upper()

        is_correct = (user_sign == target)
        xp_earned = current_drill.get("xp_reward", 15) if is_correct else 0
        self.total_xp += xp_earned

        log_entry = {
            "drill_id": current_drill["id"],
            "category": current_drill.get("category", "Alphabet Mastery"),
            "prompt": current_drill["prompt"],
            "target_letter": target,
            "learner_sign": user_sign,
            "is_correct": is_correct,
            "xp_earned": xp_earned,
            "hint": current_drill.get("hint", "")
        }
        self.drills_log.append(log_entry)
        self.current_question_index = self.current_drill_index  # compatibility alias
        self.current_drill_index += 1

        if self.is_finished():
            self.end_time = time.time()

        return is_correct

    def is_finished(self):
        """Checks if all drills in the current module have been completed."""
        return self.current_drill_index >= len(self.drills)

    def get_session_report(self):
        """Calculates total XP, accuracy percentage, time taken, and skill tier."""
        if not self.is_finished() and self.start_time is not None:
            duration = round(time.time() - self.start_time, 1)
        elif self.start_time and self.end_time:
            duration = round(self.end_time - self.start_time, 1)
        else:
            duration = 0.0

        total_drills = len(self.drills)
        correct_count = sum(1 for d in self.drills_log if d["is_correct"])
        max_xp = sum(d.get("xp_reward", 15) for d in self.drills)
        accuracy = (correct_count / total_drills * 100) if total_drills > 0 else 0.0

        if accuracy >= 85:
            rank = "🌟 Master Signer (Level 1 Certified)"
        elif accuracy >= 65:
            rank = "⚡ Skilled Signer"
        elif accuracy >= 50:
            rank = "🌱 Apprentice Signer"
        else:
            rank = "📖 Practicing Learner"

        return {
            "learner_name": self.learner_name,
            "learner_id": self.learner_id,
            "total_drills": total_drills,
            "correct_drills": correct_count,
            "total_xp": self.total_xp,
            "max_xp": max_xp,
            "accuracy_pct": round(accuracy, 1),
            "rank": rank,
            "streak_days": self.streak_days,
            "duration_seconds": duration,
            "drills_log": self.drills_log
        }

    def print_terminal_report(self):
        """Prints a clean ASCII certificate report in terminal."""
        rep = self.get_session_report()
        print("\n" + "=" * 60)
        print("         🌟 SIGNLEARN AI — SKILL MASTERY REPORT")
        print("=" * 60)
        print(f" Learner Name   : {rep['learner_name']} ({rep['learner_id']})")
        print(f" Daily Streak   : 🔥 {rep['streak_days']} Days Active")
        print(f" Session Time   : {rep['duration_seconds']} seconds")
        print("-" * 60)
        print(f" Drills Mastered: {rep['correct_drills']} / {rep['total_drills']}")
        print(f" Total XP Earned: ⭐ {rep['total_xp']} / {rep['max_xp']} XP ({rep['accuracy_pct']}%)")
        print(f" Mastery Rank   : {rep['rank']}")
        print("=" * 60)
        
        print("\n📋 Detailed Skill Feedback:")
        for idx, item in enumerate(rep["drills_log"], 1):
            status = "✅ Mastered" if item["is_correct"] else "❌ Needs Practice"
            print(f"\nDrill {idx}: {item['prompt']}")
            print(f"   You Signed: [{item['learner_sign']}] | Target: [{item['target_letter']}] -> {status}")
            print(f"   Tip: {item['hint']}")
        print("\n" + "=" * 60)


# ==============================================================================
# LIVE CAMERA ASL PRACTICE ENGINE
# ==============================================================================
def run_live_practice():
    """
    Runs the standalone OpenCV ASL practice loop.
    """
    print("🚀 Initializing SignLearn Live ASL Practice Engine...")

    drills_path = "learning_drills.json" if os.path.exists("learning_drills.json") else "asl_project/learning_drills.json"
    engine = SignLearnEngine(drills_path)
    engine.start_session(learner_name="Tochay", learner_id="SL-NCAIR-042")

    # Load Model & Detector
    label_map_file = "label_map.json" if os.path.exists("label_map.json") else "asl_project/label_map.json"
    with open(label_map_file, "r") as f:
        LABEL_MAP = json.load(f)
    IDX_TO_LABEL = {v: k for k, v in LABEL_MAP.items()}

    model = nn.Sequential(
        nn.Linear(63, 128), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, len(IDX_TO_LABEL))
    )
    weights_file = "mlp_model_final.pth" if os.path.exists("mlp_model_final.pth") else "asl_project/mlp_model_final.pth"
    model.load_state_dict(torch.load(weights_file, map_location="cpu"))
    model.eval()

    task_path = "hand_landmarker.task" if os.path.exists("hand_landmarker.task") else "asl_project/hand_landmarker.task"

    base_options = python.BaseOptions(model_asset_path=task_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3,
        min_tracking_confidence=0.3
    )
    detector = vision.HandLandmarker.create_from_options(options)

    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),
        (0, 5), (5, 6), (6, 7), (7, 8),
        (0, 9), (9, 10), (10, 11), (11, 12),
        (0, 13), (13, 14), (14, 15), (15, 16),
        (0, 17), (17, 18), (18, 19), (19, 20),
        (5, 9), (9, 13), (13, 17)
    ]

    def normalize_landmarks(raw):
        centered = raw - raw[0]
        scale = np.linalg.norm(centered[9])
        if scale < 1e-6: scale = 1e-6
        return (centered / scale).flatten()

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    WINDOW_NAME = "SignLearn AI — Interactive ASL Tutor"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_TOPMOST, 1)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    prediction_buffer = deque(maxlen=10)
    hold_sign = None
    hold_count = 0
    REQUIRED_HOLD_FRAMES = 18
    feedback_message = None
    feedback_timer = 0
    frame_count = 0
    last_landmarks = None

    print("\n🚀 SignLearn AI is running! Form the requested signs to earn XP.")

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.005)
            continue

        frame_count += 1
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        current_drill = engine.get_current_drill()

        if frame_count % 2 == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            results = detector.detect(mp_image)

            if results.hand_landmarks and len(results.hand_landmarks) > 0:
                hand = results.hand_landmarks[0]
                last_landmarks = hand
                raw = np.array([[lm.x, lm.y, lm.z] for lm in hand], dtype=np.float32)

                features = normalize_landmarks(raw)
                x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

                with torch.no_grad():
                    logits = model(x)
                    probs = torch.softmax(logits, dim=1)[0]
                    pred_idx = logits.argmax(dim=1).item()
                    pred_letter = IDX_TO_LABEL[pred_idx]
                    confidence = probs[pred_idx].item() * 100

                prediction_buffer.append(pred_letter)
                most_common = Counter(prediction_buffer).most_common(1)[0][0]

                if not engine.is_finished() and feedback_timer <= 0:
                    if confidence >= 65:
                        if most_common == hold_sign:
                            hold_count += 1
                        else:
                            hold_sign = most_common
                            hold_count = 1

                        if hold_count >= REQUIRED_HOLD_FRAMES:
                            is_correct = engine.submit_sign(hold_sign)
                            status_text = f"MASTERED! +{current_drill['xp_reward']} XP" if is_correct else f"INCORRECT (Target was: {current_drill['target_letter']})"
                            feedback_message = f"Signed [{hold_sign}] -> {status_text}"
                            feedback_timer = 25
                            hold_count = 0
                            hold_sign = None
                    else:
                        hold_count = 0
                        hold_sign = None
            else:
                last_landmarks = None
                hold_count = 0
                hold_sign = None

        if last_landmarks is not None:
            pixel_points = [(int(lm.x * w), int(lm.y * h)) for lm in last_landmarks]
            for p in pixel_points:
                cv2.circle(frame, p, 5, (0, 255, 0), -1)
            for p1, p2 in HAND_CONNECTIONS:
                if p1 < len(pixel_points) and p2 < len(pixel_points):
                    cv2.line(frame, pixel_points[p1], pixel_points[p2], (0, 220, 255), 2)

        # Header Bar
        cv2.rectangle(frame, (0, 0), (w, 65), (20, 25, 35), -1)
        header_text = f"Learner: {engine.learner_name} | Streak: 🔥 {engine.streak_days} Days | XP: ⭐ {engine.total_xp} pts"
        cv2.putText(frame, header_text, (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        if not engine.is_finished():
            # Practice Card
            d_num = engine.current_drill_index + 1
            cv2.rectangle(frame, (20, 80), (w - 20, 260), (15, 20, 30), -1)
            cv2.rectangle(frame, (20, 80), (w - 20, 260), (0, 220, 100), 2)

            d_title = f"Drill {d_num} of {len(engine.drills)} [{current_drill.get('category', 'Alphabet')}] — ⭐ +{current_drill.get('xp_reward', 15)} XP:"
            cv2.putText(frame, d_title, (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 220, 255), 2)
            cv2.putText(frame, current_drill["prompt"], (40, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
            cv2.putText(frame, f"💡 Tip: {current_drill.get('hint', '')}", (40, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 2)

            # Bottom Status Bar
            cv2.rectangle(frame, (20, h - 85), (w - 20, h - 15), (20, 25, 35), -1)
            if feedback_timer > 0:
                feedback_timer -= 1
                color = (0, 255, 0) if "MASTERED" in feedback_message else (0, 0, 255)
                cv2.putText(frame, feedback_message, (40, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            elif hold_sign and hold_count > 0:
                progress_pct = int((hold_count / REQUIRED_HOLD_FRAMES) * 100)
                msg = f"Holding Sign [{hold_sign}]... Verifying Form: {progress_pct}%"
                cv2.putText(frame, msg, (40, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
            else:
                cv2.putText(frame, f"👉 Show the sign for '{current_drill['target_letter']}' to the camera and hold steady",
                            (40, h - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)
        else:
            # Final Report Card
            rep = engine.get_session_report()
            cv2.rectangle(frame, (80, 120), (w - 80, h - 80), (15, 25, 40), -1)
            cv2.rectangle(frame, (80, 120), (w - 80, h - 80), (0, 255, 0), 3)

            cv2.putText(frame, "🌟 MODULE LEVEL 1 COMPLETED!", (w // 2 - 240, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 3)
            cv2.putText(frame, f"Learner: {rep['learner_name']} (Streak: 🔥 {rep['streak_days']} Days)", (120, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2)
            cv2.putText(frame, f"Total XP Earned: ⭐ {rep['total_xp']} / {rep['max_xp']} XP ({rep['accuracy_pct']}%)", (120, 290), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
            cv2.putText(frame, f"Mastery Rank: {rep['rank']}", (120, 340), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.putText(frame, f"Practice Time: {rep['duration_seconds']} seconds", (120, 390), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.putText(frame, "Press 'q' to close and view detailed skill breakdown in terminal.", (120, 460), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 165, 255), 2)

        cv2.imshow(WINDOW_NAME, frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    engine.print_terminal_report()


if __name__ == "__main__":
    run_live_practice()
