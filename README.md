# Falite

**F**ace **A**uthentication **L**ayer for **I**dentity and **T**rusted **E**ntry

![Python](https://img.shields.io/badge/python-3.12-blue)
![Platform](https://img.shields.io/badge/platform-Linux%20(Arch%2FEndeavourOS)-informational)

Falite is a face authentication system for Linux, built from scratch with deep-learning face embeddings. It registers up to two people, recognizes them live from a webcam, and can unlock `sudo` through **PAM** (Pluggable Authentication Modules), with the password always available as a fallback.

---

## Table of Contents

- [Features](#features)
- [How It Works](#how-it-works)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [PAM Integration](#pam-integration)
- [Security Notes](#security-notes)
- [Troubleshooting](#troubleshooting)

---

## Features

- **Guided face registration:** captures 10 samples with on-screen pose hints (straight, left, right, up, down)
- **Two-person limit:** at most 2 registered faces. To add a third, delete one first
- **File-based storage:** each person is one file, `data/<name>.npy`, where the name you type is the file name
- **Live recognition:** compares the camera face with every registered file and shows `VERIFIED` or `UNKNOWN` with a confidence percentage
- **Multi-face recognition:** every face in the frame is identified separately
- **Ambiguity guard:** a match is accepted only if the best score clearly beats the runner-up
- **Duplicate-face protection:** blocks registering the same person under two file names
- **Quality checks during registration:** rejects faces that are too small, blurry, or not alone in the frame
- **Consistent preprocessing:** registration and recognition share the same crop and embedding code
- **OpenCV interface:** header bar, corner-bracket face boxes, label tags, progress bar, status messages and live FPS
- **`config.yaml` settings:** thresholds, limits, camera index and PAM timing are set in one file
- **`sudo` login through PAM:** headless face check using `pam_exec`, with password fallback, fail-closed behavior, a root-owned install and logging to the system journal

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
2. **Embedding:** the cropped face is resized to 160x160 and passed through FaceNet, giving a 512-dimensional vector.
3. **Registration:** 10 embeddings are averaged and normalized into one reference vector, saved as `data/<name>.npy`.
4. **Recognition:** a live embedding is compared with each reference by cosine similarity. It is accepted if the score is above `threshold` and clearly higher than the runner-up.

---

## Project Structure

```
Failte/
├── register.py          # registration, storage, config loading, shared helpers and UI
├── recognize.py         # live recognition window
├── faltis_main.py       # PAM helper (exit 0 = verified)
├── install_pam.sh       # installs a root-owned copy to /opt/falite
├── config.yaml          # all settings
├── data/                # registered faces (data/<name>.npy), created automatically
└── README.md
```

`data/` is listed in `.gitignore` because it contains biometric data.

---

## Installation

Tested on **EndeavourOS** (Arch Linux). MediaPipe has no wheels for the newest Python, so this project uses **Python 3.12** in a virtual environment.

```bash
# 1. Install uv
sudo pacman -S uv

# 2. Get the project
git clone https://github.com/krishtyagi0109-pixel/Failte.git
cd Failte

# 3. Create a Python 3.12 virtual environment
uv venv --python 3.12 .venv
source .venv/bin/activate          # fish shell: source .venv/bin/activate.fish

# 4. Install dependencies
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install "mediapipe==0.10.14" "numpy<2"
uv pip install facenet-pytorch --no-deps
uv pip install pillow requests tqdm pyyaml
```

Notes:

- `mediapipe==0.10.14` is pinned because newer releases removed the `mp.solutions` API used here.
- Do not install `opencv-python` separately. MediaPipe already provides OpenCV.
- Camera access needs your user in the `video` group: `sudo usermod -aG video $USER`, then log out and back in.
- The first run downloads the VGGFace2 weights (about 107 MB). If the download fails with a gateway error, fetch the file manually into `~/.cache/torch/checkpoints/`.

---

## Usage

Run from the project folder with the virtual environment active.

### Register a face

```bash
python register.py
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

Press `X` or `ESC` to quit. The label above each face shows `VERIFIED` (green) or `UNKNOWN` (red) with a confidence percentage. The file name is not shown unless `show_name` is set to `true` in `config.yaml`.

---

## Configuration

All settings live in `config.yaml`. Any setting you remove falls back to its built-in default.

| Setting | Default | Meaning |
|---|---|---|
| `camera_index` | 0 | Which camera to use (`/dev/video0`) |
| `data_dir` | data | Folder for the registered faces |
| `registration.max_faces` | 2 | Maximum registered people |
| `registration.min_samples` | 10 | Samples needed to register |
| `registration.max_name_length` | 30 | Longest allowed file name |
| `registration.min_face_size` | 80 | Smallest accepted face, in pixels |
| `registration.blur_threshold` | 30.0 | Minimum sharpness (lower accepts blurrier frames) |
| `registration.crop_margin` | 0.20 | Extra area kept around the detected face |
| `registration.duplicate_threshold` | 0.75 | Similarity at which a new face counts as already registered |
| `recognition.threshold` | 0.75 | Minimum similarity for `VERIFIED` (try 0.65 to 0.80) |
| `recognition.margin` | 0.05 | Best match must beat the runner-up by this much |
| `recognition.show_name` | false | Show the file name above the face box |
| `recognition.show_score` | true | Show the confidence percentage |
| `pam.scan_timeout` | 8.0 | Seconds the `sudo` check looks for your face |
| `pam.required_hits` | 5 | Consecutive matching frames needed |

---

## PAM Integration

PAM is the Linux layer that handles authentication for the login screen, `su` and `sudo`. Falite plugs into `sudo` through the `pam_exec` module, which runs a program and uses its exit code.

> **Status: experimental.** Read the [security notes](#security-notes) before enabling it.

### How it works

- `faltis_main.py` runs headless (no window). It scans for up to `pam.scan_timeout` seconds and needs `pam.required_hits` consecutive verified frames.
- Exit code `0` means verified and `sudo` runs without asking for a password. Anything else (no match, camera error, missing file, unsafe permissions) exits `1`, and `sudo` asks for your password as usual. It always fails closed.
- **The registered file name must equal your Linux username.** `data/krish.npy` can only authenticate the user `krish`.
- Each attempt is written to the system journal.

### Why a root-owned install is required

`pam_exec` runs as root. If the script, the Python environment, the settings or the face data stayed in your home folder, any program running as your user could edit them, make the check always succeed, and get root through `sudo`. `install_pam.sh` copies everything to `/opt/falite`, owned by root, and the helper refuses to run if it finds files a normal user can write.

### Setup

**1. Register your face under your Linux username** (for example `krish`):

```bash
python register.py      # R, then enter: krish
```

**2. Test the helper without touching PAM:**

```bash
python faltis_main.py --test ; echo "exit code: $?"
```

It should print `RESULT: VERIFIED` and exit code `0` when you look at the camera.

**3. Install the root-owned copy:**

```bash
sudo bash install_pam.sh
sudo /opt/falite/.venv/bin/python /opt/falite/faltis_main.py --test
```

Run the installer again whenever you register or delete a face, or edit `config.yaml`.

**4. Keep a root shell open in a second terminal, then edit the sudo PAM file:**

```bash
# Terminal A (leave this open until everything works)
sudo -s
nano /etc/pam.d/sudo
```

Add this line **above the first `auth` line**:

```
auth  sufficient  pam_exec.so quiet stdout /opt/falite/.venv/bin/python /opt/falite/faltis_main.py
```

The file should look like this:

```
#%PAM-1.0
auth		sufficient	pam_exec.so quiet stdout /opt/falite/.venv/bin/python /opt/falite/faltis_main.py
auth		include		system-auth
account		include		system-auth
session		include		system-auth
```

**5. Test in a new terminal:**

```bash
sudo -k          # forget the cached sudo session
sudo true        # looks for your face; if that fails, asks for your password
```

### View the log

```bash
journalctl -t falite -n 20
```

Entries show the service, the user, and the best match score.

### Undo

In the root shell, delete the `pam_exec` line from `/etc/pam.d/sudo`. `sudo` goes back to password-only immediately.

### Rules that keep you from locking yourself out

- Always use `sufficient`, never `required`, so a camera or script failure cannot block you.
- Keep a root shell open while you edit anything in `/etc/pam.d/`.
- PAM runs after the system has booted. It does not cover the bootloader or the disk encryption (LUKS) prompt.
- Starting the model takes a few seconds, so `sudo` pauses briefly before it either succeeds or asks for the password.

---

## Security Notes

- Falite is a **learning project**, not a certified security product.
- It has **no liveness check**: it compares face embeddings only, so a photo or video of you on a phone may pass. Do not rely on it to protect anything important, and lock your screen when you leave your laptop.
- Face authentication should be a **convenience on top of a password**, never the only way in.
- Face embeddings are biometric data. Keep `data/` private. It is already listed in `.gitignore`, so do not remove it from there.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `error: externally-managed-environment` | Use `uv pip install ...` instead of `pip` inside the uv venv |
| `mp.solutions` attribute error | Install `mediapipe==0.10.14` |
| `No module named 'yaml'` | `uv pip install pyyaml` |
| Model download fails with 504 | Retry, or download the weights manually into `~/.cache/torch/checkpoints/` |
| `ImportError` mentioning `register` | Keep `register.py`, `recognize.py` and `faltis_main.py` in the same folder with these exact names |
| Camera does not open | Check `ls /dev/video*`, make sure you are in the `video` group, and that no other app is using the camera. Adjust `camera_index` in `config.yaml` |
| Window does not open on Wayland | `export QT_QPA_PLATFORM=xcb` |
| Often shown as `UNKNOWN` | Lower `recognition.threshold` slightly and watch the on-screen scores |
| `sudo` ignores your face | Check the file name equals your username, run `journalctl -t falite -n 20`, and rerun `sudo bash install_pam.sh` after any change |
