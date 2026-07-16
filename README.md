# Rock Paper Scissors — Hand Gesture Classifier

A real-time Rock Paper Scissors game that uses your webcam to detect hand gestures and play against the computer. Hand landmarks are extracted with **MediaPipe** and classified using a small **PyTorch** MLP.

▶️ # Demo video

https://github.com/user-attachments/assets/42b8de77-35c1-4c8e-b3e9-6f774aad948f

 <!-- replace with your uploaded video link/embed -->

## How it works

1. **MediaPipe Hands** detects 21 hand landmarks (x, y, z) from the webcam feed — 63 features per frame.
2. A trained **MLP classifier** (PyTorch) predicts whether the gesture is Rock, Paper, or Scissors.
3. The game shows a hand-shake / hold-still countdown, captures your gesture, and compares it against a random computer choice.

## Project structure

```
RPS/
├── game.py               # Main game — run this to play
├── collect_landmarks.py  # Captures training images and extracts landmarks
├── train.py               # Trains the MLP classifier on extracted landmarks
├── model/
│   ├── rps_model.pth      # Trained model weights
│   └── scaler.pkl         # Feature scaler used at inference time
├── assets/                # Rock/paper/scissors icons
└── requirements.txt
```

## Setup

```bash
# Clone the repo
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1      # Windows
# source .venv/bin/activate     # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Play the game

```bash
python game.py
```

Show your hand to the webcam, shake it 3 times to start the countdown, then hold your gesture steady when prompted.

### Retrain the model (optional)

If you want to collect your own gesture data and retrain:

```bash
python collect_landmarks.py   # capture images per class into data1/<class>/
python train.py                # extracts landmarks, trains MLP, saves model/
```

## Model

A simple feed-forward MLP trained on 63-dimensional hand landmark vectors:

```
Linear(63 → 128) → ReLU → Dropout(0.3)
Linear(128 → 64) → ReLU → Dropout(0.3)
Linear(64 → 3)
```

Trained with Adam optimizer, cross-entropy loss, for 50 epochs.

## Requirements

- Python 3.12
- opencv-python
- mediapipe
- torch
- numpy
- pandas
- scikit-learn

## Notes

This was originally being adapted into a Gradio web app, but that work is on hold — the current version runs as a local OpenCV window (`game.py`).
