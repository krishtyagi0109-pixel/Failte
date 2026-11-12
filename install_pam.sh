#!/usr/bin/env bash
# Installs Falite into /opt/falite, owned by root, so PAM can trust it.
#
# WHY: pam_exec runs as root. If the script, the Python environment or the
# face data live in your home folder, any program running as your user could
# edit them and turn "face login" into a free root shell. Root-owned files
# close that hole.
#
# Usage (from the project folder):   sudo bash install_pam.sh
# This script does NOT edit /etc/pam.d - you do that by hand, carefully.

set -euo pipefail

# ---- file names (change here if you rename files again) ----------------
REGISTER="register.py"        # registration + shared helpers
RECOGNIZE="recognize.py"      # live recognition (identify() is used by PAM)
PAM_HELPER="faltis_main.py"   # the script that PAM runs
# ------------------------------------------------------------------------

if [[ $EUID -ne 0 ]]; then
    echo "Run with sudo:  sudo bash install_pam.sh"
    exit 1
fi

SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="/opt/falite"
USER_NAME="${SUDO_USER:?Run this through sudo, not as a root login}"
USER_HOME="$(getent passwd "$USER_NAME" | cut -d: -f6)"
WEIGHTS_NAME="20180402-114759-vggface2.pt"
WEIGHTS="$USER_HOME/.cache/torch/checkpoints/$WEIGHTS_NAME"
REG_MODULE="${REGISTER%.py}"

# ---- sanity checks ----------------------------------------------------
for f in "$REGISTER" "$RECOGNIZE" "$PAM_HELPER" "config.yaml"; do
    [[ -f "$SRC/$f" ]] || { echo "Missing $SRC/$f"; exit 1; }
done

# The scripts import each other by module name. After a rename the imports
# must be updated too, or PAM will fail with an ImportError.
if [[ "$REG_MODULE" != "register_face" ]]; then
    if grep -q "register_face" "$SRC/$RECOGNIZE" "$SRC/$PAM_HELPER"; then
        echo "ERROR: $RECOGNIZE or $PAM_HELPER still imports 'register_face',"
        echo "but your file is now '$REGISTER'. Fix it with:"
        echo "  sed -i 's/register_face/$REG_MODULE/g' $RECOGNIZE $PAM_HELPER"
        exit 1
    fi
fi

shopt -s nullglob
faces=("$SRC"/data/*.npy)
shopt -u nullglob
if [[ ${#faces[@]} -eq 0 ]]; then
    echo "No registered faces in $SRC/data. Run $REGISTER first."
    exit 1
fi

if [[ ! -f "$SRC/data/$USER_NAME.npy" ]]; then
    echo "WARNING: there is no data/$USER_NAME.npy."
    echo "The registered file name must equal your Linux username ('$USER_NAME'),"
    echo "otherwise face login will never match. Re-register using that name."
    read -r -p "Continue anyway? [y/N] " ans
    [[ "$ans" == "y" || "$ans" == "Y" ]] || exit 1
fi

[[ -f "$WEIGHTS" ]] || { echo "Model weights not found at $WEIGHTS. Run $REGISTER once first."; exit 1; }
command -v uv >/dev/null || { echo "uv is not installed: sudo pacman -S uv"; exit 1; }

# ---- copy files -------------------------------------------------------
echo "Copying files to $DEST ..."
mkdir -p "$DEST/data" "$DEST/torch/checkpoints"
cp "$SRC/$REGISTER" "$SRC/$RECOGNIZE" "$SRC/$PAM_HELPER" "$SRC/config.yaml" "$DEST/"
cp "${faces[@]}" "$DEST/data/"
cp "$WEIGHTS" "$DEST/torch/checkpoints/$WEIGHTS_NAME"

# ---- root-owned Python environment ------------------------------------
if [[ ! -x "$DEST/.venv/bin/python" ]]; then
    echo "Creating Python 3.12 environment ..."
    uv venv --python 3.12 "$DEST/.venv"
fi

echo "Installing dependencies (this can take a few minutes) ..."
PY="$DEST/.venv/bin/python"
uv pip install --python "$PY" torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python "$PY" "mediapipe==0.10.14" "numpy<2"
uv pip install --python "$PY" facenet-pytorch --no-deps
uv pip install --python "$PY" pillow requests tqdm pyyaml

# ---- lock down permissions --------------------------------------------
chown -R root:root "$DEST"
chmod -R go-w "$DEST"
chmod 755 "$DEST" "$DEST/data"
chmod 644 "$DEST"/data/*.npy

echo
echo "Installed to $DEST (owned by root)."
echo
echo "Test it now (it should print VERIFIED when you look at the camera):"
echo "  sudo $PY $DEST/$PAM_HELPER --test"
echo
echo "PAM line for /etc/pam.d/sudo (add it ABOVE the first 'auth' line):"
echo "  auth  sufficient  pam_exec.so quiet stdout $PY $DEST/$PAM_HELPER"
echo
echo "When you change faces (register/delete), run this installer again."
