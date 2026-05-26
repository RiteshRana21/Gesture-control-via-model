"""
=============================================================
  GESTURE DATA COLLECTOR
  Captures hand gesture images from webcam for training.
=============================================================
  GESTURES:
    0 - idle         (no hand — point camera at empty desk)
    1 - play_pause   (open palm — all 5 fingers spread wide)
    2 - next_track   (two fingers — index + middle pointing UP)
    3 - prev_track   (fist — all fingers fully curled closed)
    4 - volume_up    (thumbs up — only thumb up, others curled)
    5 - volume_down  (thumbs down — thumb pointing down)

  USAGE:
    python3 collect_data.py
    Press [0-5] to select a gesture class
    Press SPACE to start/stop capturing
    Press Q to quit
=============================================================
"""

import cv2
import os
import time

GESTURES = {
    "0": ("idle",        "No hand"),
    "1": ("play_pause",  "open Palm"),
    "2": ("next_track",  "Point Right"),
    "3": ("prev_track",  "Point Left"),
    "4": ("volume_up",   "Thumbs UP"),
    "5": ("volume_down", "Thumbs DOWN "),
}

SAVE_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset")
IMG_SIZE  = (64, 64)
CAPTURE_N = 300    # images per session
DELAY_MS  = 50     # ms between captures


def count_existing(class_name):
    path = os.path.join(SAVE_DIR, class_name)
    os.makedirs(path, exist_ok=True)
    return len([f for f in os.listdir(path) if f.endswith(".jpg")])


def collect():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    current_class = None
    capturing     = False
    count         = 0
    last_time     = 0

    print("\n=== GESTURE DATA COLLECTOR ===")
    print("Keys: [0-5] select gesture | [SPACE] start/stop | [Q] quit\n")
    for k, (name, hint) in GESTURES.items():
        existing = count_existing(name)
        print(f"  [{k}] {name:<15} — {hint}  (saved: {existing})")
    print()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame   = cv2.flip(frame, 1)
        display = frame.copy()

        # ── ROI box ───────────────────────────────────────────────
        h, w = frame.shape[:2]
        x1, y1 = w // 4, h // 8
        x2, y2 = 3 * w // 4, 7 * h // 8
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(display, "Keep hand inside box",
                    (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # ── UI info ───────────────────────────────────────────────
        if current_class:
            class_name, hint = GESTURES[current_class]
            existing = count_existing(class_name)
            cv2.putText(display, f"Class: {class_name}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(display, hint, (10, 58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (200, 200, 200), 1)
            cv2.putText(display, f"Total saved: {existing}  |  Session: {count}/{CAPTURE_N}",
                        (10, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)
            status = "CAPTURING..." if capturing else "Press SPACE to start"
            color  = (0, 0, 255) if capturing else (0, 200, 255)
            cv2.putText(display, status, (10, 108),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
        else:
            cv2.putText(display, "Press [0-5] to select a gesture",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.imshow("Gesture Collector", display)

        # ── Auto capture ──────────────────────────────────────────
        now = time.time() * 1000
        if capturing and current_class and (now - last_time) >= DELAY_MS:
            class_name = GESTURES[current_class][0]
            save_path  = os.path.join(SAVE_DIR, class_name)
            existing   = count_existing(class_name)
            roi        = frame[y1:y2, x1:x2]
            roi_small  = cv2.resize(roi, IMG_SIZE)
            filename   = os.path.join(save_path, f"{existing:05d}.jpg")
            cv2.imwrite(filename, roi_small)
            count    += 1
            last_time = now
            if count >= CAPTURE_N:
                capturing = False
                count     = 0
                print(f"[✓] Done! {CAPTURE_N} images saved for '{class_name}'")

        # ── Key handling ──────────────────────────────────────────
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif chr(key) in GESTURES:
            current_class = chr(key)
            capturing     = False
            count         = 0
            name, hint    = GESTURES[current_class]
            print(f"\n[→] Selected: {name}")
            print(f"    Gesture : {hint}")
            print(f"    Saved   : {count_existing(name)} images so far")
        elif key == ord(" "):
            if current_class:
                capturing = not capturing
                count     = 0
                last_time = 0
                state     = "STARTED" if capturing else "STOPPED"
                print(f"[{state}] {GESTURES[current_class][0]}")

    cap.release()
    cv2.destroyAllWindows()

    print("\n=== Collection Summary ===")
    for k, (name, _) in GESTURES.items():
        print(f"  {name:<15} : {count_existing(name)} images")
    print("\n[Done] Run train.py next!")


if __name__ == "__main__":
    collect()