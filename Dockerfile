FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    DATA_DIR=/home/user/app/data \
    FASTEMBED_CACHE_PATH=/home/user/.cache/fastembed

# Non-root user (uid 1000 is what Hugging Face Spaces expects)
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=user:user . .
RUN mkdir -p /home/user/app/data /home/user/.cache/fastembed && chown -R user:user /home/user
USER user

# Bake the embedding model into the image so the first visitor doesn't wait for a download.
RUN python -c "from fastembed import TextEmbedding; import os; TextEmbedding('BAAI/bge-small-en-v1.5', cache_dir=os.environ['FASTEMBED_CACHE_PATH'])" || echo "model pre-download skipped"

EXPOSE 7860
CMD ["python", "app.py"]
