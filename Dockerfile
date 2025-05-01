# FROM python:3.13-slim
FROM python:3.12-slim-bullseye
# FROM python:alpine

# ENV PYTHONDONTWRITEBYTECODE 1
# ENV PYTHONUNBUFFERED 1

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    build-essential \
    python3-dev \
    libatlas-base-dev \
    libffi-dev \
    && apt-get clean

WORKDIR /app

# RUN apt-get update && apt-get install -y \
RUN apt-get update && apt-get upgrade -y && apt-get clean \
    # fonts-liberation \
    # fonts-noto \
    # fonts-dejavu \
    # fontconfig \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt -i https://pypi.org/simple/

# RUN apt-get update && apt-get install -y \
#     fontconfig \
#     fonts-dejavu \
#     fonts-liberation \
#     ttf-mscorefonts-installer \
#     && fc-cache -f -v \
#     && apt-get clean

COPY . .

RUN apt-get update && apt-get install -y libicu-dev

RUN apt-get install -y fontconfig
RUN apt-get install -y fonts-freefont-ttf

EXPOSE 8000 8501

# CMD if [ "$APP_MODE" = "fastapi" ]; then \
#         uvicorn api.routes:app --host 0.0.0.0 --port 8000; \
#     else \
#         chainlit run ui/ui.py --host 0.0.0.0 --port 8501; \
#     fi

#!/bin/sh
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Use JSON form to avoid shell form issues
CMD ["/app/entrypoint.sh"]
