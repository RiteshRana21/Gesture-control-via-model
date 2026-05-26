"""
=============================================================
  REAL-TIME GESTURE -> SPOTIFY CONTROLLER
  Pure MediaPipe landmark-based gesture detection. No CNN.
=============================================================
  GESTURES:
    open palm    -> play / pause  (all 5 fingers spread open)
    point right  -> next track    (only index finger up, pointing right)
    point left   -> previous track(only index finger up, pointing left)
    thumbs up    -> volume +10    (only thumb up, others curled)
    thumbs down  -> volume -10    (only thumb down, others curled)

  USAGE:
    python3 run.py
    Press Q to quit.
=============================================================
"""

import os
import sys
import time
import collections
import urllib.request
import ssl

import cv2
import numpy as np

# ── MediaPipe ─────────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    MODEL_TASK_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
    if not os.path.exists(MODEL_TASK_PATH):
        print("[MediaPipe] Downloading hand landmarker model (~15 MB)...")
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode    = ssl.CERT_NONE
        with urllib.request.urlopen(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            context=ctx
        ) as r, open(MODEL_TASK_PATH, "wb") as f:
            f.write(r.read())
        print("[MediaPipe] Downloaded!")

    _mp_options = mp_vision.HandLandmarkerOptions(
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_TASK_PATH),
        num_hands    = 1,
        min_hand_detection_confidence = 0.6,
        min_hand_presence_confidence  = 0.6,
        min_tracking_confidence       = 0.5,
    )
    hands_detector = mp_vision.HandLandmarker.create_from_options(_mp_options)
    print("[MediaPipe] Hand detector loaded ✓")

except ImportError:
    print("[ERROR] mediapipe not installed. Run: pip3 install mediapipe")
    sys.exit(1)

# ── Spotify ───────────────────────────────────────────────────────────────────
try:
    import spotipy
    from spotipy.oauth2 import SpotifyOAuth
except ImportError:
    print("[ERROR] spotipy not installed. Run: pip3 install spotipy")
    sys.exit(1)

# ── Config ────────────────────────────────────────────────────────────────────
COOLDOWN_SEC  = 2.0
SMOOTH_FRAMES = 10
VOLUME_STEP   = 10

SPOTIFY_SCOPE = (
    "user-modify-playback-state "
    "user-read-playback-state"
)

COLOURS = {
    "idle":         (120, 120, 120),
    "play_pause":   (0,   220,   0),
    "next_track":   (255, 180,   0),
    "prev_track":   (0,   180, 255),
    "volume_up":    (0,   255, 200),
    "volume_down":  (200,  80, 255),
}

CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),(0,17)
]


# ── Spotify wrapper ───────────────────────────────────────────────────────────
class SpotifyController:
    def __init__(self):
        self.sp = spotipy.Spotify(auth_manager=SpotifyOAuth(
            client_id     = os.getenv("SPOTIPY_CLIENT_ID"),
            client_secret = os.getenv("SPOTIPY_CLIENT_SECRET"),
            redirect_uri  = os.getenv("SPOTIPY_REDIRECT_URI", "http://localhost:8888/callback"),
            scope         = SPOTIFY_SCOPE,
            open_browser  = True,
        ))
        self._last_volume = 50

    def _active_device(self):
        devices = self.sp.devices().get("devices", [])
        active  = [d for d in devices if d["is_active"]]
        return active[0] if active else (devices[0] if devices else None)

    def toggle_play_pause(self):
        try:
            state = self.sp.current_playback()
            if state and state.get("is_playing"):
                self.sp.pause_playback()
                return "Paused"
            else:
                device = self._active_device()
                kwargs = {"device_id": device["id"]} if device else {}
                self.sp.start_playback(**kwargs)
                return "Playing"
        except Exception as e:
            return f"[!] {e}"

    def next_track(self):
        try:
            self.sp.next_track()
            return "Next track"
        except Exception as e:
            return f"[!] {e}"

    def prev_track(self):
        try:
            self.sp.previous_track()
            return "Previous track"
        except Exception as e:
            return f"[!] {e}"

    def change_volume(self, delta):
        try:
            state   = self.sp.current_playback()
            vol     = state["device"]["volume_percent"] if state else self._last_volume
            new_vol = max(0, min(100, vol + delta))
            self.sp.volume(new_vol)
            self._last_volume = new_vol
            return f"Vol {'+' if delta > 0 else ''}{delta}%  ->  {new_vol}%"
        except Exception as e:
            return f"[!] {e}"


# ── Landmark indices ──────────────────────────────────────────────────────────
WRIST      = 0
THUMB_MCP  = 2;  THUMB_IP   = 3;  THUMB_TIP  = 4
INDEX_MCP  = 5;  INDEX_PIP  = 6;  INDEX_TIP  = 8
MIDDLE_PIP = 10; MIDDLE_TIP = 12
RING_PIP   = 14; RING_TIP   = 16
PINKY_PIP  = 18; PINKY_TIP  = 20


# ── Gesture classifier ────────────────────────────────────────────────────────
def finger_up(lm, tip, pip):
    return lm[tip].y < lm[pip].y

def classify_gesture(lm):
    index_up  = finger_up(lm, INDEX_TIP,  INDEX_PIP)
    middle_up = finger_up(lm, MIDDLE_TIP, MIDDLE_PIP)
    ring_up   = finger_up(lm, RING_TIP,   RING_PIP)
    pinky_up  = finger_up(lm, PINKY_TIP,  PINKY_PIP)

    thumb_up      = lm[THUMB_TIP].y < lm[THUMB_MCP].y - 0.04
    thumb_down    = lm[THUMB_TIP].y > lm[THUMB_IP].y + 0.04
    fingers_curled = not index_up and not middle_up and not ring_up and not pinky_up
    all_up         = index_up and middle_up and ring_up and pinky_up

    # Open palm -> play/pause
    if all_up:
        return "play_pause"

    # Thumbs up -> volume up
    if thumb_up and fingers_curled:
        return "volume_up"

    # Thumbs down -> volume down
    if thumb_down and fingers_curled and not thumb_up:
        return "volume_down"

    # Index pointing -> next/prev
    if index_up and not middle_up and not ring_up and not pinky_up:
        diff = lm[INDEX_TIP].x - lm[WRIST].x
        if diff > 0.08:
            return "next_track"
        elif diff < -0.08:
            return "prev_track"

    return "idle"


# ── Draw hand skeleton ────────────────────────────────────────────────────────
def draw_landmarks(display, lm, x1, y1, x2, y2):
    roi_h = y2 - y1
    roi_w = x2 - x1
    pts = [(int(l.x * roi_w) + x1, int(l.y * roi_h) + y1) for l in lm]
    for a, b in CONNECTIONS:
        cv2.line(display, pts[a], pts[b], (0, 200, 0), 1)
    for px, py in pts:
        cv2.circle(display, (px, py), 4, (0, 255, 0), -1)


# ── Main loop ─────────────────────────────────────────────────────────────────
def run():
    from pathlib import Path
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if "=" in line and not line.startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    try:
        spotify = SpotifyController()
        print("[Spotify] Connected ✓")
    except Exception as e:
        print(f"[Spotify] Connection failed: {e}")
        spotify = None

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    gesture_buffer = collections.deque(maxlen=SMOOTH_FRAMES)
    last_action_t  = {}
    status_msg     = "Show your hand!"
    status_time    = time.time()

    ACTIONS = {
        "play_pause":  lambda sp: sp.toggle_play_pause(),
        "next_track":  lambda sp: sp.next_track(),
        "prev_track":  lambda sp: sp.prev_track(),
        "volume_up":   lambda sp: sp.change_volume(+VOLUME_STEP),
        "volume_down": lambda sp: sp.change_volume(-VOLUME_STEP),
    }

    HINTS = {
        "play_pause":  "open palm",
        "next_track":  "point right ->",
        "prev_track":  "<- point left",
        "volume_up":   "thumbs up",
        "volume_down": "thumbs down",
        "idle":        "idle",
    }

    print("\n[Running] Press Q to quit.")
    print("open palm=play/pause | point right=next | point left=prev | thumbs up=vol+ | thumbs down=vol-\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        display = frame.copy()
        h, w    = frame.shape[:2]

        x1, y1 = w // 4, h // 8
        x2, y2 = 3 * w // 4, 7 * h // 8
        roi     = frame[y1:y2, x1:x2]
        roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # Detect hand
        mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=roi_rgb)
        mp_result    = hands_detector.detect(mp_image)
        hand_present = len(mp_result.hand_landmarks) > 0

        if not hand_present:
            gesture_buffer.append("idle")
            smooth_gesture = "idle"
            cv2.rectangle(display, (x1, y1), (x2, y2), (100, 100, 100), 2)
            cv2.rectangle(display, (x1, y1 - 30), (x2, y1), (100, 100, 100), -1)
            cv2.putText(display, "No hand detected", (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        else:
            lm = mp_result.hand_landmarks[0]
            draw_landmarks(display, lm, x1, y1, x2, y2)

            gesture = classify_gesture(lm)
            gesture_buffer.append(gesture)
            smooth_gesture = collections.Counter(gesture_buffer).most_common(1)[0][0]

            # Trigger Spotify
            now = time.time()
            if (smooth_gesture != "idle"
                    and smooth_gesture in ACTIONS
                    and spotify is not None
                    and (now - last_action_t.get(smooth_gesture, 0)) >= COOLDOWN_SEC):
                result                        = ACTIONS[smooth_gesture](spotify)
                status_msg                    = result
                status_time                   = now
                last_action_t[smooth_gesture] = now
                print(f"[Action] {smooth_gesture}  ->  {result}")

            colour = COLOURS.get(smooth_gesture, (200, 200, 200))
            cv2.rectangle(display, (x1, y1), (x2, y2), colour, 2)
            hint  = HINTS.get(smooth_gesture, "")
            label = f"{smooth_gesture}  ({hint})"
            cv2.rectangle(display, (x1, y1 - 30), (x2, y1), colour, -1)
            cv2.putText(display, label, (x1 + 5, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2)

        # Status message
        now = time.time()
        if now - status_time < 3.0:
            cv2.putText(display, status_msg, (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 180), 2)

        # Legend
        legend = [
            ("open palm = play/pause",  COLOURS["play_pause"]),
            ("point right = next",      COLOURS["next_track"]),
            ("point left = prev",       COLOURS["prev_track"]),
            ("thumbs up = vol+",        COLOURS["volume_up"]),
            ("thumbs down = vol-",      COLOURS["volume_down"]),
        ]
        for i, (name, col) in enumerate(legend):
            cv2.circle(display, (12, 20 + i * 22), 6, col, -1)
            cv2.putText(display, name, (22, 25 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, col, 1)

        cv2.imshow("Gesture Spotify Controller", display)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hands_detector.close()
    print("\n[Done] Controller stopped.")


if __name__ == "__main__":
    run()