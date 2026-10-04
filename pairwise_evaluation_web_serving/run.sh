#!/usr/bin/env bash
# Launch the HTTPS pairwise evaluation app (behind Nginx).
#
# Required env vars:
#   ADMIN_API_KEY   — secret key for admin endpoints (set a strong random value)
#
# Optional env vars:
#   PORT                       — Uvicorn port (default: 7860, must match nginx/eval.conf)
#   OPENAI_API_KEY             — enables OpenAI content moderation
#   PROLIFIC_COMPLETION_CODE   — Prolific completion code shown on the done page
#   PROLIFIC_COMPLETION_URL    — overrides the auto-built completion URL
#
# Usage:
#   ADMIN_API_KEY="$(openssl rand -hex 32)" ./run.sh

set -euo pipefail
cd "$(dirname "$0")"

if [[ -z "${PROLIFIC_COMPLETION_CODE:-}" ]]; then
    echo "NOTE: PROLIFIC_COMPLETION_CODE is not set. The done page will not show a completion code."
fi

if [[ -z "${ADMIN_API_KEY:-}" ]]; then
    echo "WARNING: ADMIN_API_KEY is not set. Admin endpoints are unprotected."
    echo "  Set it with: export ADMIN_API_KEY=\$(openssl rand -hex 32)"
fi

exec python -m uvicorn app:app \
    --host 127.0.0.1 \
    --port "${PORT:-7860}" \
    --proxy-headers \
    --forwarded-allow-ips='*' \
    --no-access-log \
    "$@"
