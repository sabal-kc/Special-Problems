#!/usr/bin/env bash
set -euo pipefail

# Idempotent setup script for Amazon Linux 2023 EC2.
# It installs deps, prepares a venv, downloads NLTK data,
# installs systemd unit, and starts/enables the service.
#
# Usage:
#   bash services/setup_ec2_service.sh
#   bash services/setup_ec2_service.sh --repo-dir "$HOME/Special-Problems" --port 8000

REPO_DIR="${HOME}/Special-Problems"
PORT="8000"
SERVICE_NAME="traditional-sentiment"
SERVICE_USER="${USER}"

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
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

SERVICE_FILE="${REPO_DIR}/services/traditional_sentiment_service.py"
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

echo "Creating/updating virtualenv..."
python3 -m venv "${VENV_DIR}"
"${VENV_DIR}/bin/pip" install --upgrade pip
"${VENV_DIR}/bin/pip" install -r "${REQ_FILE}"

echo "Ensuring NLTK vader_lexicon is present..."
"${VENV_DIR}/bin/python" - <<'PY'
import nltk
nltk.download("vader_lexicon", quiet=True)
print("vader_lexicon ready")
PY

echo "Installing systemd unit: ${UNIT_FILE}"
sudo tee "${UNIT_FILE}" >/dev/null <<EOF
[Unit]
Description=Traditional Sentiment Service
After=network.target

[Service]
Type=simple
User=${SERVICE_USER}
WorkingDirectory=${REPO_DIR}
ExecStart=${VENV_DIR}/bin/python ${SERVICE_FILE} --host 0.0.0.0 --port ${PORT}
Restart=always
RestartSec=3

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
curl -s "http://127.0.0.1:${PORT}/health" || true
echo
echo "Done. Service will auto-start on boot."
