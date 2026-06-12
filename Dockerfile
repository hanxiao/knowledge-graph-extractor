FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    numpy \
    python-multipart \
    pypdf \
    sentence-transformers \
    peft \
    einops \
    torch --extra-index-url https://download.pytorch.org/whl/cpu

COPY app.py .
COPY jobs.py .
COPY jina-corpus.zip .
COPY favicon.ico .
COPY assets/ ./assets/

EXPOSE 3000

CMD ["python", "app.py"]
