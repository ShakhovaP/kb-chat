#!/bin/sh
if [ "$APP_MODE" = "fastapi" ]; then
    exec uvicorn api.routes:app --host 0.0.0.0 --port 8000
    # exec python -m api.routes --host 0.0.0.0 --port 8000
else
    exec chainlit run ui/ui.py --host 0.0.0.0 --port 8501
fi
