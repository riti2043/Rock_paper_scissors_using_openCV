import cv2
import mediapipe as mp
import torch
import torch.nn as nn
import numpy as np
import pickle
import random
import time

# Load model
class RPS_MLP(nn.Module):
    def __init__(self):
        super(RPS_MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(63, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)
        )
    
    def forward(self, x):
        return self.model(x)

model = RPS_MLP()
model.load_state_dict(torch.load("model/rps_model.pth"))
model.eval()

with open("model/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Labels
CLASSES = ["Rock", "Paper", "Scissors"]
COMPUTER_CHOICES = ["Rock", "Paper", "Scissors"]

def get_winner(player, computer):
    if player == computer:
        return "Draw"
    wins = {"Rock": "Scissors", "Paper": "Rock", "Scissors": "Paper"}
    return "You Win" if wins[player] == computer else "You Lose"

def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

def predict_gesture(landmarks):
    landmarks = np.array(landmarks).reshape(1, -1)
    landmarks = scaler.transform(landmarks)
    tensor = torch.tensor(landmarks, dtype=torch.float32)
    with torch.no_grad():
        output = model(tensor)
        predicted = torch.argmax(output, dim=1).item()
    return CLASSES[predicted]

# Game state
STATE = "waiting"   # waiting → countdown → result
countdown_start = None
player_gesture = None
computer_gesture = None
outcome = None
shake_positions = []
last_shake_time = time.time()
shake_count = 0

cap = cv2.VideoCapture(0)
WINDOW_W, WINDOW_H = 1280, 480

print("Game started — show your hand and shake 3 times!")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # Create canvas
    canvas = np.zeros((WINDOW_H, WINDOW_W, 3), dtype=np.uint8)
    canvas[:] = (30, 30, 30)

    # Player side — right half
    player_frame = cv2.resize(frame, (580, 440))
    canvas[20:460, 680:1260] = player_frame

    # VS text in center
    cv2.putText(canvas, "VS", (610, 260), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

    # Computer side — left half placeholder
    cv2.rectangle(canvas, (20, 20), (600, 460), (60, 60, 60), -1)
    cv2.putText(canvas, "Computer", (180, 250), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (180, 180, 180), 2)

    current_time = time.time()

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        mp_draw.draw_landmarks(canvas[20:460, 680:1260], hand_landmarks, mp_hands.HAND_CONNECTIONS)

        wrist_y = hand_landmarks.landmark[0].y

        if STATE == "waiting":
            # Shake detection
            shake_positions.append(wrist_y)
            if len(shake_positions) > 10:
                shake_positions.pop(0)

            if len(shake_positions) == 10:
                motion = max(shake_positions) - min(shake_positions)
                if motion > 0.1 and current_time - last_shake_time > 0.4:
                    shake_count += 1
                    last_shake_time = current_time
                    print(f"Shake {shake_count}")

            if shake_count >= 3:
                STATE = "countdown"
                countdown_start = current_time
                shake_count = 0
                shake_positions = []
                computer_gesture = random.choice(COMPUTER_CHOICES)

        elif STATE == "countdown":
            elapsed = current_time - countdown_start
            remaining = 2 - elapsed

            if remaining > 0:
                cv2.putText(canvas, f"Hold still... {remaining:.1f}s",
                            (400, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                # Capture gesture
                landmarks = extract_landmarks(hand_landmarks)
                player_gesture = predict_gesture(landmarks)
               
                outcome = get_winner(player_gesture, computer_gesture)
                STATE = "result"

    if STATE == "waiting":
        cv2.putText(canvas, f"Shake your fist 3 times! ({shake_count}/3)",
                    (350, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)

    elif STATE == "result":
        # Show computer gesture text
        cv2.putText(canvas, computer_gesture, (180, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)

        # Show player gesture
        cv2.putText(canvas, f"You: {player_gesture}",
                    (700, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        # Show outcome
        color = (0, 255, 0) if outcome == "You Win" else (0, 0, 255) if outcome == "You Lose" else (255, 255, 0)
        cv2.putText(canvas, outcome, (490, 440),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)

        # Reset after 3 seconds
        if current_time - countdown_start > 5:
            STATE = "waiting"
            player_gesture = None
            computer_gesture = None
            outcome = None

    cv2.imshow("Rock Paper Scissors", canvas)

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()