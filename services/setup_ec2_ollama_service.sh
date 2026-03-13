#!/usr/bin/env bash
set -euo pipefail

# Setup script for Ollama sentiment service on Amazon Linux 2023 EC2.
# Installs Ollama, pulls the model, installs Python deps, and runs the service.
#
# Usage:
#   bash services/setup_ec2_ollama_service.sh
#   bash services/setup_ec2_ollama_service.sh --repo-dir "$HOME/Special-Problems" --port 8000

REPO_DIR="${HOME}/Special-Problems"
PORT="8000"
SERVICE_NAME="ollama-sentiment"
SERVICE_USER="${USER}"
OLLAMA_MODEL="llama3.2:3b-instruct-q3_K_S"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --repo-dir)
      REPO_DIR="$2"
      shift 2
      ;;
    --port)
      PORT="$2"
      shift 2
      ;;
    --service-user)
      SERVICE_USER="$2"
      shift 2
      ;;
    --model)
      OLLAMA_MODEL="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

SERVICE_FILE="${REPO_DIR}/services/ollama_sentiment_service.py"
REQ_FILE="${REPO_DIR}/services/requirements.txt"
VENV_DIR="${REPO_DIR}/.venv"
UNIT_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

if [[ ! -f "${SERVICE_FILE}" ]]; then
  echo "Service file not found: ${SERVICE_FILE}"
  exit 1
fi

if [[ ! -f "${REQ_FILE}" ]]; then
  echo "Requirements file not found: ${REQ_FILE}"
  exit 1
fi

echo "Installing OS packages..."
sudo dnf install -y python3 python3-pip

echo "Installing Ollama..."
if command -v ollama &>/dev/null; then
  echo "Ollama already installed."
else
  curl -fsSL https://ollama.com/install.sh | sh
fi

echo "Configuring Ollama parallelism (OLLAMA_NUM_PARALLEL=2)..."
sudo mkdir -p /etc/systemd/system/ollama.service.d
sudo tee /etc/systemd/system/ollama.service.d/override.conf >/dev/null <<EOF
[Service]
Environment=OLLAMA_NUM_PARALLEL=2
EOF

echo "Ensuring Ollama service is running..."
sudo systemctl daemon-reload
if systemctl is-active --quiet ollama 2>/dev/null; then
  echo "Ollama service already running, restarting to apply configuration..."
  sudo systemctl restart ollama 2>/dev/null || OLLAMA_NUM_PARALLEL=2 ollama serve &
else
  sudo systemctl start ollama 2>/dev/null || OLLAMA_NUM_PARALLEL=2 ollama serve &
fi
sleep 3

echo "Pulling model: ${OLLAMA_MODEL} (may take a while)..."
ollama pull "${OLLAMA_MODEL}"

echo "Creating/updating virtualenv..."
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install -r "${REQ_FILE}"

echo "Installing systemd unit: ${UNIT_FILE}"
sudo tee "${UNIT_FILE}" >/dev/null <<EOF
[Unit]
Description=Ollama Sentiment Service
After=network.target ollama.service
Wants=ollama.service

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${REPO_DIR}
ExecStart=${VENV_DIR}/bin/python ${SERVICE_FILE} --host 0.0.0.0 --port ${PORT}
Restart=always
RestartSec=3
Environment=HOME=${HOME}
Environment=OLLAMA_MODEL=${OLLAMA_MODEL}

[Install]
WantedBy=multi-user.target
EOF

echo "Reloading and enabling service..."
sudo systemctl daemon-reload
sudo systemctl enable --now "${SERVICE_NAME}"

echo
echo "Service status:"
sudo systemctl --no-pager --full status "${SERVICE_NAME}" || true
echo
echo "Health check:"
sleep 2
curl -s "http://127.0.0.1:${PORT}/health" || true
echo
echo "Done. Service will auto-start on boot."
