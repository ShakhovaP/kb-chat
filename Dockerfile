FROM python:3.11-slim

# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000 8501

CMD if [ "$APP_MODE" = "fastapi" ]; then \
        uvicorn api.routes:app --host 0.0.0.0 --port 8000; \
    else \
        chainlit run ui/ui.py --host 0.0.0.0 --port 8501; \
    fi