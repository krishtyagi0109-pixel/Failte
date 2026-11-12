"""
Controls: X or ESC to quit.
"""

import time


import cv2

import mediapipe as mp

import numpy as np

from facenet_pytorch import InceptionResnetV1

from register import (
    CAMERA_INDEX,
    CFG,
    DANGER,
    DATA_DIR,
    DEVICE,
    MIN_FACE_SIZE,
    MUTED,
    SUCCESS,
    WARNING,
    crop_face,
    draw_chip,
    draw_corner_box,
    draw_footer,
    draw_header,
    draw_toast,
    get_embedding,
    load_db,
)

# ----------------------------- Config ---------------------------------
_rec = CFG["recognition"]
THRESHOLD = _rec["threshold"]      # min cosine similarity to accept a match
MARGIN = _rec["margin"]            # best match must beat the runner-up by this
SHOW_NAME = _rec["show_name"]      # show the file name above the box
SHOW_SCORE = _rec["show_score"]    # show confidence (%) in the label


def identify(embedding: np.ndarray, db: dict[str, np.ndarray]):
    """Return (name or None, best_similarity)."""
    scores = {name: float(np.dot(ref, embedding)) for name, ref in db.items()}
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    best_name, best_score = ranked[0]

    if best_score < THRESHOLD:
        return None, best_score


    if len(ranked) > 1 and best_score - ranked[1][1] < MARGIN:
        return None, best_score

    return best_name, best_score


def main() -> None:
    db = load_db()
    if not db:
        print(f"No registered faces found in {DATA_DIR}/. Run register.py first.")
        return

    model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.6
    )

    cap = cv2.VideoCapture(CAMERA_INDEX)
    prev_time = time.time()
    fps = 0.0

    try:
        if not cap.isOpened():
            print("Cannot open webcam.")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read from webcam.")
                break

            frame = cv2.flip(frame, 1)  
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            results = detector.process(rgb)
            face_found = False

            if results.detections:
                for det in results.detections:
                    b = det.location_data.relative_bounding_box
                    x1, y1 = max(0, int(b.xmin * w)), max(0, int(b.ymin * h))
                    x2 = min(w, int((b.xmin + b.width) * w))
                    y2 = min(h, int((b.ymin + b.height) * h))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    face_found = True
                    box = (x1, y1, x2, y2)

                    if min(x2 - x1, y2 - y1) < MIN_FACE_SIZE:
                        draw_corner_box(frame, box, WARNING)
                        draw_chip(frame, "MOVE CLOSER", box, WARNING)
                        continue

                    face_rgb = crop_face(rgb, box)
                    if face_rgb.size == 0:
                        continue
                    emb = get_embedding(model, face_rgb)
                    if emb is None:
                        continue

                    name, score = identify(emb, db)
                    color = SUCCESS if name else DANGER

                    label = "VERIFIED" if name else "UNKNOWN"
                    if name and SHOW_NAME:
                        label = f"VERIFIED  {name}"
                    if SHOW_SCORE:
                        label += f"  {max(score, 0) * 100:.0f}%"

                    draw_corner_box(frame, box, color)
                    draw_chip(frame, label, box, color)

            now = time.time()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev_time, 1e-6))
            prev_time = now

            draw_header(frame, "FACE RECOGNITION", f"Enrolled: {len(db)}    FPS: {fps:.0f}")
            if not face_found:
                draw_toast(frame, "No face detected", MUTED)
            draw_footer(frame, "[X] or [ESC] Quit")

            cv2.imshow("Face Recognition", frame)
            if cv2.waitKey(1) & 0xFF in (ord("x"), 27):
                break
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
