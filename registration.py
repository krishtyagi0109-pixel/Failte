import re

from pathlib import Path

import cv2

import mediapipe as mp

import numpy as np

import torch

from facenet_pytorch import InceptionResnetV1

# ----------------------------- Config ---------------------------------
DATA_DIR = Path("data")
MAX_FACES = 2
MIN_SAMPLES = 10
MAX_NAME_LEN = 30
NAME_PATTERN = re.compile(r"^[A-Za-z0-9_-]+$")

MIN_FACE_SIZE = 80          # px, smallest accepted face box side
BLUR_THRESHOLD = 30.0       # Laplacian variance; lower = blurrier
CROP_MARGIN = 0.20          # extra context around the detected box
DUPLICATE_THRESHOLD = 0.75  # cosine similarity above this = "same person"

DEVICE = torch.device("cpu")

POSE_HINTS = [
    "Look straight at the camera",
    "Turn your head slightly left",
    "Turn your head slightly right",
    "Tilt your head up slightly",
    "Tilt your head down slightly",
]

# ----------------------------- UI theme (BGR) -------------------------
FONT = cv2.FONT_HERSHEY_DUPLEX
AA = cv2.LINE_AA
WHITE = (240, 240, 240)
PANEL = (28, 28, 28)
ACCENT = (200, 170, 20)
SUCCESS = (110, 200, 70)
WARNING = (0, 190, 255)
DANGER = (80, 80, 235)
MUTED = (170, 170, 170)

HEADER_H = 44
FOOTER_H = 36


def put_text(frame, text, org, scale=0.55, color=WHITE, thickness=1):
    cv2.putText(frame, text, org, FONT, scale, color, thickness, AA)


def text_width(text, scale=0.55, thickness=1) -> int:
    return cv2.getTextSize(text, FONT, scale, thickness)[0][0]


def overlay_rect(frame, p1, p2, color=PANEL, alpha=0.70):
    """Semi-transparent filled rectangle."""
    x1, y1 = max(0, p1[0]), max(0, p1[1])
    x2, y2 = min(frame.shape[1], p2[0]), min(frame.shape[0], p2[1])
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    tint = np.full_like(roi, color)
    frame[y1:y2, x1:x2] = cv2.addWeighted(roi, 1 - alpha, tint, alpha, 0)


def draw_header(frame, title: str, right: str = ""):
    w = frame.shape[1]
    overlay_rect(frame, (0, 0), (w, HEADER_H))
    cv2.line(frame, (0, HEADER_H), (w, HEADER_H), ACCENT, 2, AA)
    put_text(frame, title, (16, 29), 0.7, WHITE, 1)
    if right:
        put_text(frame, right, (w - text_width(right, 0.5) - 16, 28), 0.5, MUTED)


def draw_footer(frame, text: str):
    h, w = frame.shape[:2]
    overlay_rect(frame, (0, h - FOOTER_H), (w, h))
    x = (w - text_width(text, 0.5)) // 2
    put_text(frame, text, (x, h - 12), 0.5, MUTED)


def draw_toast(frame, text: str, color):
    """Centered status message just above the footer."""
    h, w = frame.shape[:2]
    tw = text_width(text, 0.6)
    x1, y2 = (w - tw) // 2 - 16, h - FOOTER_H - 14
    overlay_rect(frame, (x1, y2 - 34), (x1 + tw + 32, y2), PANEL, 0.80)
    cv2.line(frame, (x1, y2 - 34), (x1, y2), color, 3, AA)
    put_text(frame, text, (x1 + 16, y2 - 11), 0.6, color)


def draw_corner_box(frame, box, color, thickness=2):
    x1, y1, x2, y2 = box
    length = max(14, int(min(x2 - x1, y2 - y1) * 0.22))
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                             (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (cx, cy), (cx + dx * length, cy), color, thickness, AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * length), color, thickness, AA)


def draw_chip(frame, text: str, box, color):
    """Small filled label attached to the top-left of a box."""
    x1, y1, _, y2 = box
    tw = text_width(text, 0.5)
    ch = 26
    top = y1 - ch - 4
    if top < HEADER_H + 4:          # no room above -> place below the box
        top = y2 + 4
    overlay_rect(frame, (x1, top), (x1 + tw + 20, top + ch), color, 0.90)
    put_text(frame, text, (x1 + 10, top + 18), 0.5, (20, 20, 20))


def draw_progress(frame, current: int, total: int, y: int):
    w = frame.shape[1]
    x1, x2 = 16, w - 16
    ratio = min(1.0, current / total)
    overlay_rect(frame, (x1, y), (x2, y + 8), (70, 70, 70), 0.9)
    if ratio > 0:
        overlay_rect(frame, (x1, y), (x1 + int((x2 - x1) * ratio), y + 8), SUCCESS, 1.0)


# ----------------------------- Storage --------------------------------
def face_path(name: str) -> Path:
    return DATA_DIR / f"{name}.npy"


def load_db() -> dict[str, np.ndarray]:
    """Load every data/*.npy file. The file name (without .npy) is the key."""
    db: dict[str, np.ndarray] = {}
    if not DATA_DIR.exists():
        return db
    for path in sorted(DATA_DIR.glob("*.npy")):
        try:
            vec = np.load(path).astype(np.float32).reshape(-1)
            if vec.shape != (512,):
                raise ValueError(f"unexpected shape {vec.shape}")
            vec = normalize(vec)
            if vec is None:
                raise ValueError("zero vector")
            db[path.stem] = vec
        except (OSError, ValueError) as e:
            print(f"Warning: skipping {path.name} ({e})")
    return db


def save_face(name: str, reference: np.ndarray) -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = face_path(name)
    np.save(path, reference.astype(np.float32))
    return path


# ----------------------------- Helpers --------------------------------
def normalize(vec: np.ndarray) -> np.ndarray | None:
    norm = np.linalg.norm(vec)
    return None if norm == 0 else vec / norm


def crop_face(rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    """Square crop around the box with a small margin."""
    h, w = rgb.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    half = max(x2 - x1, y2 - y1) * (1 + CROP_MARGIN) / 2
    nx1, ny1 = int(max(0, cx - half)), int(max(0, cy - half))
    nx2, ny2 = int(min(w, cx + half)), int(min(h, cy + half))
    return rgb[ny1:ny2, nx1:nx2]


def is_sharp(face_rgb: np.ndarray) -> bool:
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var() >= BLUR_THRESHOLD


def get_embedding(model, face_rgb: np.ndarray) -> np.ndarray | None:
    face = cv2.resize(face_rgb, (160, 160)).astype(np.float32)
    face = (face - 127.5) / 128.0
    tensor = torch.from_numpy(face.transpose(2, 0, 1)).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        emb = model(tensor).cpu().numpy()[0]
    return normalize(emb)


def ask_file_name(existing: dict[str, np.ndarray]) -> str | None:
    print("\nThis name will be the saved file name, e.g.  data/<name>.npy")
    while True:
        name = input("File name (letters, numbers, _ or -; blank to cancel): ").strip()
        if not name:
            return None
        if len(name) > MAX_NAME_LEN:
            print(f"Too long (max {MAX_NAME_LEN} characters).")
            continue
        if not NAME_PATTERN.match(name):
            print("Use only letters, numbers, underscore (_) or hyphen (-).")
            continue
        if name.lower() in (n.lower() for n in existing):
            print("A file with that name already exists. Choose another.")
            continue
        return name


# ----------------------------- Capture --------------------------------
def capture_reference(model, file_name: str) -> np.ndarray | None:
    """Open webcam, collect MIN_SAMPLES embeddings, return the mean reference."""
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.6
    )
    cap = cv2.VideoCapture(0)
    embeddings: list[np.ndarray] = []
    toast, toast_color = "", WHITE
    window = "Face Registration"

    try:
        if not cap.isOpened():
            print("Cannot open webcam.")
            return None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Cannot read from webcam.")
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]

            boxes = []
            results = detector.process(rgb)
            if results.detections:
                for det in results.detections:
                    b = det.location_data.relative_bounding_box
                    x1, y1 = max(0, int(b.xmin * w)), max(0, int(b.ymin * h))
                    x2 = min(w, int((b.xmin + b.width) * w))
                    y2 = min(h, int((b.ymin + b.height) * h))
                    if x2 > x1 and y2 > y1:
                        boxes.append((x1, y1, x2, y2))

            # Face boxes
            if len(boxes) == 1:
                draw_corner_box(frame, boxes[0], SUCCESS)
                draw_chip(frame, "FACE DETECTED", boxes[0], SUCCESS)
            else:
                for bx in boxes:
                    draw_corner_box(frame, bx, WARNING)
                if len(boxes) > 1:
                    draw_chip(frame, "MULTIPLE FACES", boxes[0], WARNING)

            # HUD
            draw_header(frame, "FACE REGISTRATION", f"File: {file_name}.npy")
            put_text(frame, f"Samples  {len(embeddings)} / {MIN_SAMPLES}", (16, 72), 0.55, WHITE)
            hint = POSE_HINTS[len(embeddings) % len(POSE_HINTS)]
            put_text(frame, hint, (w - text_width(hint, 0.55) - 16, 72), 0.55, ACCENT)
            draw_progress(frame, len(embeddings), MIN_SAMPLES, 82)
            if toast:
                draw_toast(frame, toast, toast_color)
            draw_footer(frame, "[SPACE] Capture     [X] Finish     [ESC] Cancel")

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF

            if key == 27:  # ESC
                print("Registration cancelled.")
                return None
            if key == ord("x"):
                break

            if key == ord(" "):
                toast = ""

                if len(boxes) != 1:
                    toast, toast_color = "Exactly one face must be visible", DANGER
                    continue

                x1, y1, x2, y2 = boxes[0]
                if min(x2 - x1, y2 - y1) < MIN_FACE_SIZE:
                    toast, toast_color = "Face too small - move closer", DANGER
                    continue

                face_rgb = crop_face(rgb, boxes[0])
                if face_rgb.size == 0:
                    toast, toast_color = "Invalid crop - try again", DANGER
                    continue

                if not is_sharp(face_rgb):
                    toast, toast_color = "Image too blurry - hold still", DANGER
                    continue

                emb = get_embedding(model, face_rgb)
                if emb is None:
                    toast, toast_color = "Could not read face - try again", DANGER
                    continue

                embeddings.append(emb)
                toast, toast_color = f"Sample {len(embeddings)} captured", SUCCESS
                print(f"Captured sample {len(embeddings)}/{MIN_SAMPLES}")
    finally:
        cap.release()
        detector.close()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    if len(embeddings) < MIN_SAMPLES:
        print(f"Registration cancelled. Capture at least {MIN_SAMPLES} samples "
              f"(got {len(embeddings)}).")
        return None

    return normalize(np.mean(np.stack(embeddings), axis=0))


# ----------------------------- Actions --------------------------------
def list_faces(db: dict[str, np.ndarray]) -> None:
    print(f"\nRegistered faces ({len(db)}/{MAX_FACES}):")
    if not db:
        print("  (none)")
    for i, name in enumerate(db, 1):
        print(f"  {i}. {face_path(name)}")


def delete_face(db: dict[str, np.ndarray]) -> None:
    if not db:
        print("Nothing to delete.")
        return
    list_faces(db)
    choice = input("Enter the file name or number to delete (blank to cancel): ").strip()
    if not choice:
        return

    names = list(db)
    if choice.isdigit() and 1 <= int(choice) <= len(names):
        target = names[int(choice) - 1]
    else:
        target = next((n for n in names if n.lower() == choice.lower().removesuffix(".npy")), None)

    if target is None:
        print("No matching file found.")
        return

    if input(f"Delete {face_path(target)}? (y/n): ").strip().lower() == "y":
        face_path(target).unlink(missing_ok=True)
        del db[target]
        print(f"Deleted {target}.npy")


def register_face(model, db: dict[str, np.ndarray]) -> None:
    if len(db) >= MAX_FACES:
        print(f"\nMaximum of {MAX_FACES} faces already registered.")
        print("Delete one first if you want to add a new face.")
        return

    file_name = ask_file_name(db)
    if file_name is None:
        print("Cancelled.")
        return

    reference = capture_reference(model, file_name)
    if reference is None:
        return

    # Prevent registering the same person twice under different file names
    for other_name, other_ref in db.items():
        sim = float(np.dot(reference, other_ref))
        if sim >= DUPLICATE_THRESHOLD:
            print(f"This face looks the same as '{other_name}.npy' "
                  f"(similarity {sim:.2f}). Not saved.")
            return

    path = save_face(file_name, reference)
    db[file_name] = reference
    print(f"\nRegistration successful! Saved to {path}")


# ----------------------------- Main -----------------------------------
def main() -> None:
    print("Loading face model...")
    model = InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)
    db = load_db()

    while True:
        list_faces(db)
        print("\n[R] Register new face   [D] Delete a face   [Q] Quit")
        choice = input("Choose: ").strip().lower()

        if choice == "r":
            register_face(model, db)
        elif choice == "d":
            delete_face(db)
        elif choice == "q":
            print("Bye!")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()
