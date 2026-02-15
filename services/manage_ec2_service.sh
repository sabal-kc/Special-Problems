#!/usr/bin/env bash
set -euo pipefail

SERVICE_NAME="traditional-sentiment"

ACTION="${1:-status}"

case "${ACTION}" in
  start)
    sudo systemctl start "${SERVICE_NAME}"
    ;;
  stop)
    sudo systemctl stop "${SERVICE_NAME}"
    ;;
  restart)
    sudo systemctl restart "${SERVICE_NAME}"
    ;;
  status)
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
