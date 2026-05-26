# 🎵 Gesture Spotify Controller — ML from Scratch

Control Spotify with hand gestures using a **custom CNN trained entirely from scratch** (no MediaPipe, no pretrained weights).

---

## 🧠 How It Works

```
Webcam → ROI Crop (64×64) → GestureCNN → Gesture Class → Spotify API
```

### Model Architecture (GestureCNN)
- **4 × ConvBlock**: Conv2d → BatchNorm → ReLU → MaxPool → Dropout2d
- **Global Average Pool**: removes spatial dimensions
- **FC Classifier**: 256 → 128 → 6 classes
- **~500K trainable parameters** — fast, real-time inference

### Gestures & Actions
| Gesture | Action |
|---------|--------|
| ✊ Closed fist / 🖐 Open palm | Play / Pause |
| 👉 Point right | Next track |
| 👈 Point left | Previous track |
| 👍 Thumbs up | Volume +10% |
| 👎 Thumbs down | Volume -10% |
| 😐 Idle / neutral | No action |

---

## 🚀 Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Spotify credentials
1. Go to [Spotify Developer Dashboard](https://developer.spotify.com/dashboard)
2. Create a new app → copy **Client ID** and **Client Secret**
3. Add `http://localhost:8888/callback` as a **Redirect URI** in your app settings
4. Fill in `.env`:
```
SPOTIPY_CLIENT_ID=your_client_id
SPOTIPY_CLIENT_SECRET=your_client_secret
SPOTIPY_REDIRECT_URI=http://localhost:8888/callback
```

### 3. Collect training data
```bash
python data_collection/collect_data.py
```
- Press **[0–5]** to select a gesture class
- Press **[SPACE]** to start/stop auto-capture
- Aim for **300+ images per class** (1800 total)
- Vary your hand position, lighting, and distance!

### 4. Train the model
```bash
python train.py
```
- Trains for up to 50 epochs with early stopping
- Best model saved to `models/gesture_cnn_best.pth`
- Typical training time: ~5–15 min on CPU, ~1–2 min on GPU

### 5. Run the controller
```bash
python run.py
```
- Keep your hand inside the green box
- Hold a gesture steady for ~1 second to trigger the action
- Press **Q** to quit

---

## 📁 Project Structure

```
gesture_spotify/
├── data_collection/
│   └── collect_data.py      # Webcam data capture tool
├── dataset/
│   ├── idle/                # ~300 images each
│   ├── play_pause/
│   ├── next_track/
│   ├── prev_track/
│   ├── volume_up/
│   └── volume_down/
├── models/
│   ├── gesture_cnn.py       # CNN architecture (from scratch)
│   ├── gesture_cnn_best.pth # Saved after training
│   └── training_history.json
├── train.py                 # Full training pipeline
├── run.py                   # Real-time inference + Spotify
├── requirements.txt
└── .env                     # Your Spotify credentials
```

---

## 💡 Tips for Better Accuracy

| Tip | Why |
|-----|-----|
| Collect in multiple lighting conditions | Prevents overfitting to one environment |
| Vary hand distance from camera | Makes model robust to scale |
| Include both left and right hand | Doubles your training diversity |
| Capture 400–500 images per class | More data = better generalization |
| Keep background simple at first | Reduces noise during early training |

---

## 🔧 Tuning

Edit top of `train.py` to adjust:
```python
EPOCHS      = 50      # increase if still improving
BATCH_SIZE  = 32      # reduce if RAM issues
LR          = 1e-3    # try 3e-4 for fine-tuning
PATIENCE    = 8       # early stopping patience
```

Edit top of `run.py` to adjust:
```python
CONF_THRESHOLD = 0.75   # raise if false triggers, lower if too slow
COOLDOWN_SEC   = 1.5    # seconds between repeated actions
SMOOTH_FRAMES  = 10     # majority-vote window (higher = smoother)
VOLUME_STEP    = 10     # % volume change per gesture
```

---

## 📊 Expected Results

With 300 images/class and default settings:
- **Val Accuracy**: ~90–96% after training
- **Inference speed**: ~30 FPS on CPU, ~60+ FPS on GPU

---

Built with PyTorch · OpenCV · Spotipy
