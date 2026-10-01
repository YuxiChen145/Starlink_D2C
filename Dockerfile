# Starlink D2C — Measurement Explorer
# Builds a code-only image; mount your data at run time:
#   docker build -t starlink-d2c .
#   docker run -p 8501:8501 -v "/abs/path/to/data:/data" starlink-d2c
# where /abs/path/to/data contains "QP CSV", "PRISM CSV", "Sigcap CSV" and
# (optionally) "TLE".

FROM python:3.12-slim

WORKDIR /app

# Install dependencies first so the layer caches across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code (data is mounted, not baked in — see .dockerignore).
COPY . .

# Data + cache live on a mounted volume at /data.
ENV QP_DATA_DIR="/data/QP CSV" \
    PRISM_DATA_DIR="/data/PRISM CSV" \
    SIGCAP_DATA_DIR="/data/Sigcap CSV" \
    TLE_DATA_DIR="/data/TLE" \
    QP_CACHE_DIR="/data/.cache/qpviz"

EXPOSE 8501

# Basic container healthcheck against Streamlit's status endpoint.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health',timeout=3).status==200 else 1)" || exit 1

CMD ["streamlit", "run", "app.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true", "--browser.gatherUsageStats=false"]
