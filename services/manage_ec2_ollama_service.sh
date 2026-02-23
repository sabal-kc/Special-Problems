#!/usr/bin/env bash
set -euo pipefail

# Manage the ollama-sentiment systemd service.
# Usage: ./manage_ec2_ollama_service.sh {start|stop|restart|status|logs|disable}

SERVICE_NAME="ollama-sentiment"

ACTION="${1:-status}"

case "${ACTION}" in
  start)
    sudo systemctl start ollama 2>/dev/null || true
    sudo systemctl start "${SERVICE_NAME}"
    ;;
  stop)
    sudo systemctl stop "${SERVICE_NAME}"
    ;;
  restart)
    sudo systemctl restart ollama 2>/dev/null || true
    sudo systemctl restart "${SERVICE_NAME}"
    ;;
  status)
    echo "=== Ollama (LLM runtime) ==="
    systemctl is-active ollama 2>/dev/null || echo "not managed by systemd"
    echo
    echo "=== ${SERVICE_NAME} ==="
    sudo systemctl --no-pager --full status "${SERVICE_NAME}"
    ;;
  logs)
    sudo journalctl -u "${SERVICE_NAME}" -f
    ;;
  disable)
    sudo systemctl disable --now "${SERVICE_NAME}"
    ;;
  *)
    echo "Usage: $0 {start|stop|restart|status|logs|disable}"
    exit 1
    ;;
esac
