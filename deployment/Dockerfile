# Pneumonia X-Ray Classifier -- Streamlit deployment image
#
# IMPORTANT: build context must be the REPO ROOT (the folder that contains
# both `src/` and `deployment/`), not the `deployment/` folder itself, since
# this image needs the shared `src/preprocessing.py` module that also lives
# outside `deployment/`. GitHub Codespaces' devcontainer.json in this repo is
# already configured for this ("context": ".."); building by hand looks like:
#
#   docker build -f deployment/Dockerfile -t pneumonia-classifier .
#   docker run -p 8501:8501 pneumonia-classifier

FROM python:3.11-slim

# System libs needed by opencv-python-headless / pydicom
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (better layer caching)
COPY deployment/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Shared preprocessing package (single source of truth with the training
# notebook) + the app itself + the serialized model artifacts.
COPY src ./src
COPY deployment/app.py .
COPY deployment/model_artifacts ./model_artifacts

EXPOSE 8501

HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
