#!/bin/sh
# Starts Autosampler at http://localhost:8765
cd "$(dirname "$0")"
if [ ! -d .venv ]; then
  python3.11 -m venv .venv || python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
(sleep 1.5 && open "http://localhost:8765") &
exec .venv/bin/uvicorn server.app:app --port 8765
