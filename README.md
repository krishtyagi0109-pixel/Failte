# Falite

**F**ace **A**uthentication **L**ayer for **I**dentity and **T**rusted **E**ntry


Falite is a face authentication system for Linux, built from scratch with deep-learning face embeddings. It registers up to two people, recognizes them live from a webcam, and is designed to plug into **PAM** (Pluggable Authentication Modules) so a face match can unlock `sudo` and the login screen, with the password as a fallback.

> **Project status:** the registration and recognition core is working. Liveness detection, PAM integration, logging, and evaluation are on the roadmap below.

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [PAM Integration](#pam-integration)
- [Anti-Spoofing and Liveness](#anti-spoofing-and-liveness)
- [Evaluation](#evaluation)
- [Security Notes and Limitations](#security-notes-and-limitations)
- [Roadmap](#roadmap)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Features

### Available now

- **Live face registration** with an on-screen guided capture of 10 samples and pose hints (straight, left, right, up, down)
- **Two-person limit:** at most 2 registered faces. To add a third, one of the two must be deleted first
- **File-based storage:** each person is one file, `data/<name>.npy`, where the name you type is the file name
- **Live recognition** that compares the camera face against every registered file and shows `VERIFIED` or `UNKNOWN` with a confidence percentage
- **Ambiguity guard:** a match is accepted only if the best score clearly beats the runner-up
- **Duplicate-face protection:** blocks registering the same person under two file names
- **Quality checks during registration:** rejects faces that are too small, blurry, or not alone in the frame
- **Consistent preprocessing:** registration and recognition share the same crop and embedding code
- **Professional OpenCV interface:** header bar, corner-bracket face boxes, label tags, progress bar, status messages, and live FPS
- **Multi-face recognition:** every face in the frame is identified separately

### In development (see [Roadmap](#roadmap))

- **Liveness detection** with blink detection and a random challenge (blink twice, turn left, and so on)
- **PAM integration** for `sudo` and the login screen, with password fallback
- **Access log** in CSV format, with snapshots of unknown faces
- **Attempt limiting and lockout** after repeated failures
- **`config.yaml`** for all settings, with no hardcoded thresholds
- **Evaluation toolkit** reporting false accept and false reject rates and a threshold analysis
- **Modular code structure**, `requirements.txt`, tests, and a demo GIF

---

## How It Works

```
Camera frame
    |
    v
Face detection (MediaPipe) --> crop with margin --> quality checks
    |
    v
Embedding (FaceNet InceptionResnetV1, VGGFace2) --> 512-d unit vector
    |
    v
Cosine similarity against data/*.npy
    |
    v
Threshold + margin decision --> VERIFIED / UNKNOWN
```

1. **Detection:** MediaPipe finds the face box in each frame.
2. **Embedding:** the cropped face is resized to 160x160 and passed through FaceNet, giving a 512-dimensional vector that represents the face.
3. **Registration:** 10 embeddings are averaged and normalized into one reference vector, saved as `data/<name>.npy`.
4. **Recognition:** a live embedding is compared with each reference by cosine similarity. It is accepted if the score is above `THRESHOLD` and clearly higher than the runner-up.

---

## Project Structure

Current:

```
falite/
├── register_face.py     # registration, storage, shared helpers and UI
├── recognize.py         # live recognition
├── data/                # registered faces (data/<name>.npy), created automatically
└── README.md
```

Planned:

```
falite/
├── falite/
│   ├── detector.py      # face detection
│   ├── embedder.py      # FaceNet embeddings
│   ├── liveness.py      # blink and challenge-response checks
│   ├── storage.py       # reading and writing data/*.npy
│   ├── ui.py            # OpenCV drawing helpers
│   └── logger.py        # access log and snapshots
├── pam/
│   └── falite_pam.py    # PAM helper (exit 0 = match)
├── eval/
│   └── evaluate.py      # FAR/FRR and threshold analysis
├── config.yaml
├── requirements.txt
├── tests/
└── README.md
```

---

## Installation

Tested on **EndeavourOS** (Arch Linux). MediaPipe has no wheels for the newest Python, so this project uses **Python 3.12** in a virtual environment.

```bash
# 1. Install uv
sudo pacman -S uv

# 2. Get the project
git clone https://github.com/<your-username>/falite.git
cd falite

# 3. Create a Python 3.12 virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate          # fish shell: source .venv/bin/activate.fish

# 4. Install dependencies
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install "mediapipe==0.10.14" "numpy<2"
uv pip install facenet-pytorch --no-deps
uv pip install pillow requests tqdm
```

Notes:

- `mediapipe==0.10.14` is pinned because newer releases removed the `mp.solutions` API used here.
- Do not install `opencv-python` separately. MediaPipe already provides OpenCV.
- Camera access needs your user in the `video` group: `sudo usermod -aG video $USER`, then log out and back in.
- The first run downloads the VGGFace2 weights (about 107 MB). If the download fails with a gateway error, fetch the file manually into `~/.cache/torch/checkpoints/`.

---

## Usage

Always run from the project folder with the virtual environment active.

### Register a face

```bash
python register_face.py
```

| Key (menu) | Action |
|---|---|
| `R` | Register a new face. You enter a file name, then capture 10 samples |
| `D` | Delete a registered face (frees a slot) |
| `Q` | Quit |

| Key (camera window) | Action |
|---|---|
| `SPACE` | Capture a sample |
| `X` | Finish and save (needs at least 10 samples) |
| `ESC` | Cancel |

The name you type becomes the file, for example `krish` gives `data/krish.npy`. Use only letters, numbers, `_` and `-`.

### Run recognition

```bash
python recognize.py
```

Press `X` or `ESC` to quit. The label above each face shows `VERIFIED` (green) or `UNKNOWN` (red) with a confidence percentage. The file name is not shown by default (`SHOW_NAME = False`).

---

## Configuration

Settings currently live at the top of the scripts and will move to `config.yaml`.

| Setting | Default | Meaning |
|---|---|---|
| `MAX_FACES` | 2 | Maximum registered people |
| `MIN_SAMPLES` | 10 | Samples needed to register |
| `MIN_FACE_SIZE` | 80 px | Smallest accepted face |
| `BLUR_THRESHOLD` | 30.0 | Minimum sharpness (Laplacian variance) |
| `DUPLICATE_THRESHOLD` | 0.75 | Similarity at which a new face counts as already registered |
| `THRESHOLD` | 0.75 | Minimum similarity to accept a match (tune between 0.65 and 0.80) |
| `MARGIN` | 0.05 | Best match must beat the runner-up by this much |
| `SHOW_NAME` | False | Show the file name above the face box |

Planned `config.yaml`:

```yaml
data_dir: data
max_faces: 2
camera_index: 0

registration:
  min_samples: 10
  min_face_size: 80
  blur_threshold: 30.0
  duplicate_threshold: 0.75

recognition:
  threshold: 0.75
  margin: 0.05
  show_name: false
  show_score: true

liveness:
  enabled: true
  blink_required: 2
  challenge: random

security:
  max_failed_attempts: 5
  lockout_seconds: 60
```

---

## PAM Integration

PAM (Pluggable Authentication Modules) is the Linux layer that handles authentication for the login screen, lock screen, `sudo` and `su`. Falite plans to act as one PAM step through `pam_exec`.

> **Status: planned.** The PAM helper is not written yet. The design below is the intended approach.

**Design**

- A helper script, `pam/falite_pam.py`, opens the camera, runs liveness and recognition, and exits `0` on a verified match and non-zero otherwise.
- It is added to a PAM service file as `sufficient`, so a face match lets you in and a failure falls back to the password:

```
# /etc/pam.d/sudo   (planned example)
auth  sufficient  pam_exec.so quiet /path/to/falite/.venv/bin/python /path/to/falite/pam/falite_pam.py
auth  include     system-auth
```

**Safety rules**

- Use `sufficient`, never `required`, so a camera or script failure cannot lock you out.
- Keep a root terminal open while editing anything in `/etc/pam.d/`.
- Test with `sudo` first. Add the login screen only after that works.
- Use absolute paths. PAM runs as root without your virtual environment or working directory.
- A timeout makes the module fail fast and fall back to the password.

**Scope:** PAM runs after the system has booted. It cannot protect the bootloader or disk encryption (LUKS) prompt.

---

## Anti-Spoofing and Liveness

A face system that only compares embeddings can be fooled by a photo or a phone screen. Planned defenses, in layers:

| Layer | Idea | Status |
|---|---|---|
| Blink detection | Eye aspect ratio from MediaPipe Face Mesh landmarks | Planned |
| Challenge-response | Random action (blink twice, turn left, smile) so a pre-recorded video cannot match | Planned |
| Head-pose movement | Check that landmarks move in a natural 3D way | Planned |
| Passive anti-spoof model | Texture, moire and glare analysis per frame | Considered |
| Rate limiting | Lockout after repeated failures, with snapshots logged | Planned |

---

## Evaluation

An evaluation script will measure how well the system actually works, using consenting volunteers.

| Metric | Meaning | Result |
|---|---|---|
| FAR (false accept rate) | Wrong person accepted | TBD |
| FRR (false reject rate) | Right person rejected | TBD |
| Photo attack | Printed photo or phone photo accepted | TBD |
| Video replay attack | Phone video accepted | TBD |
| Lighting robustness | Dim, bright, backlit | TBD |

Planned output: a plot of FAR and FRR against `THRESHOLD` to choose an operating point.

---

## Security Notes and Limitations

- Falite is a **learning and portfolio project**, not a certified security product.
- With a normal webcam, high-quality video replays, 3D masks and deepfakes can still defeat software-only liveness checks.
- Face authentication should be a **convenience on top of a password**, never the only way in.
- Face embeddings are biometric data. Keep `data/` private, and do not commit it to GitHub (see `.gitignore` below).

Recommended `.gitignore`:

```
.venv/
data/
__pycache__/
*.pyc
logs/
```

---

## Roadmap

- [x] Face detection and embedding pipeline (MediaPipe + FaceNet)
- [x] Registration with guided capture and quality checks
- [x] Two-person limit with delete and re-register
- [x] File-based storage (`data/<name>.npy`)
- [x] Live recognition with confidence and ambiguity guard
- [x] Professional OpenCV interface
- [ ] Blink detection
- [ ] Random challenge-response liveness
- [ ] `config.yaml` settings
- [ ] Access log and unknown-face snapshots
- [ ] Failed-attempt limit and lockout
- [ ] PAM helper for `sudo`
- [ ] PAM for the login screen
- [ ] Evaluation script and results
- [ ] Modular package structure and tests
- [ ] `requirements.txt`, screenshots and demo GIF

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `error: externally-managed-environment` | Use `uv pip install ...` instead of `pip` inside the uv venv |
| `mp.solutions` attribute error | Install `mediapipe==0.10.14` |
| Model download fails with 504 | Retry, or download the weights manually into `~/.cache/torch/checkpoints/` |
| `ImportError: cannot import name ...` from `register_face` | `register_face.py` is an old version, replace it with the current one |
| Camera does not open | Check `ls /dev/video*`, make sure you are in the `video` group, and that no other app is using the camera |
| Window does not open on Wayland | `export QT_QPA_PLATFORM=xcb` |
| Often shown as `UNKNOWN` | Lower `THRESHOLD` slightly and watch the on-screen scores |

---

## License

MIT License. Add a `LICENSE` file to the repository.

## Author

Krish
