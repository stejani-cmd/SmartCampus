# Use a slim Python image
FROM python:3.11-slim

# System deps for building Python wheels & git (some ML libs need it)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential git curl && \
    rm -rf /var/lib/apt/lists/*

# Prevent Python from writing .pyc, and force unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Workdir
WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt ./
# Your code imports huggingface_hub, add it if it’s not listed
RUN python - <<'PY'
reqs = open('requirements.txt').read().strip().splitlines()
extra = ['huggingface_hub']
with open('requirements.txt','w') as f:
    have = {r.split('==')[0] for r in reqs}
    for r in reqs:
        f.write(r+'\n')
    for e in extra:
        if e not in have:
            f.write(e+'\n')
PY

# Install Python deps (CPU-only Torch is fine; this can take a while the first time)
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the app
COPY . .

# Spaces sets PORT (usually 7860). Listen on it.
ENV PORT=7860
EXPOSE 7860

# Cache models (optional): mount to /root/.cache in compose for persistence
# VOLUME ["/root/.cache"]

# Default command: uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "${PORT}"]
