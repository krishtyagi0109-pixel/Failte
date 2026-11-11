#!/usr/bin/env python3

import os

import pwd

import stat

import sys

import syslog

import time

from pathlib import Path

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"


if (BASE / "torch").exists():
    os.environ["TORCH_HOME"] = str(BASE / "torch")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("GLOG_minloglevel", "2")

# ----------------------------- Config ---------------------------------
CAMERA_INDEX = 0
SCAN_TIMEOUT = 8.0    
REQUIRED_HITS = 5      


def log(msg: str) -> None:
    syslog.openlog("falite", syslog.LOG_PID, syslog.LOG_AUTH)
    syslog.syslog(syslog.LOG_INFO, msg)


def say(msg: str) -> None:
    print(f"Falite: {msg}", flush=True)


def is_unsafe(path: Path) -> bool:
    """True if a non-root user could modify this path."""
    st = path.stat()
    return st.st_uid != 0 or bool(st.st_mode & (stat.S_IWGRP | stat.S_IWOTH))


def check_permissions() -> str | None:
    """When running as root, refuse to trust files a normal user can edit.

    Otherwise anyone who can edit these files could make this script always
    succeed and gain root through sudo.
    """
    targets = [BASE, Path(__file__).resolve(), DATA, *DATA.glob("*.npy")]
    for p in targets:
        if is_unsafe(p):
            return f"unsafe permissions on {p}"
    return None


def verify(user: str) -> bool:
    import cv2
    import mediapipe as mp

    from facenet_pytorch import InceptionResnetV1

    import register as rf
    from recognize import identify

    rf.DATA_DIR = DATA                       # absolute path
    db = rf.load_db()
    if user not in db:
        log(f"no registered face file for user '{user}'")
        say(f"no face registered for '{user}'")
        return False

    model = InceptionResnetV1(pretrained="vggface2").eval().to(rf.DEVICE)
    detector = mp.solutions.face_detection.FaceDetection(
        model_selection=0, min_detection_confidence=0.6
    )
    cap = cv2.VideoCapture(CAMERA_INDEX)

    try:
        if not cap.isOpened():
            log("camera could not be opened")
            say("camera unavailable")
            return False

        say("looking for your face...")
        hits = 0
        best = 0.0
        deadline = time.time() + SCAN_TIMEOUT

        while time.time() < deadline:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.05)
                continue

            frame = cv2.flip(frame, 1)       
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w = frame.shape[:2]
            matched = False

            results = detector.process(rgb)
            if results.detections and len(results.detections) == 1:
                b = results.detections[0].location_data.relative_bounding_box
                x1, y1 = max(0, int(b.xmin * w)), max(0, int(b.ymin * h))
                x2 = min(w, int((b.xmin + b.width) * w))
                y2 = min(h, int((b.ymin + b.height) * h))

                if x2 > x1 and y2 > y1 and min(x2 - x1, y2 - y1) >= rf.MIN_FACE_SIZE:
                    face = rf.crop_face(rgb, (x1, y1, x2, y2))
                    emb = rf.get_embedding(model, face) if face.size else None
                    if emb is not None:
                        name, score = identify(emb, db)
                        best = max(best, score)
                        matched = (name == user)

            hits = hits + 1 if matched else 0
            if hits >= REQUIRED_HITS:
                log(f"face verified for '{user}' (best score {best:.2f})")
                say("face verified")
                return True

        log(f"face NOT verified for '{user}' (best score {best:.2f})")
        say("face not recognized, use your password")
        return False
    finally:
        cap.release()
        detector.close()


def main() -> int:
    test_mode = "--test" in sys.argv
    user = os.environ.get("PAM_USER")
    if not user and test_mode:
        user = os.environ.get("SUDO_USER")   
    if not user:
        user = pwd.getpwuid(os.getuid()).pw_name

    if os.geteuid() == 0:
        problem = check_permissions()
        if problem:
            log(problem)
            say("install is not secure, skipping face login")
            return 1

    try:
        ok = verify(user)
    except Exception as exc:                 
        log(f"error: {exc!r}")
        say("error, skipping face login")
        if test_mode:
            import traceback
            traceback.print_exc()
        return 1

    if test_mode:
        print("RESULT:", "VERIFIED" if ok else "NOT VERIFIED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
