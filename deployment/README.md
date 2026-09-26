# Deployment — Streamlit + Docker + GitHub Codespaces

This folder satisfies the Final report's **Model Deployment** rubric item: build a Streamlit
app, package it in Docker, push it to a repository, and run it in GitHub Codespaces with a
forwarded URL for a live inference.

## 0. Prerequisite

Run the training notebook (`notebooks/Capstone_Pneumonia_Classification.ipynb`) through
**Section 7 (Model Deployment)**. That section copies two files into
`deployment/model_artifacts/`:

- `best_model.keras` — the serialized best-performing model
- `model_config.json` — input size, channel count, class names, and headline test metrics

The Dockerfile copies this folder directly into the image, so it must be populated first.

## 1. Push to a repository

From the repo root (the folder containing both `src/` and `deployment/`):

```bash
git init   # if not already a repo
git add src/ deployment/ .devcontainer/
git commit -m "Add pneumonia classifier deployment"
git remote add origin <your-repo-url>
git push -u origin main
```

## 2. Run locally (optional, quick check before pushing)

```bash
cd deployment
pip install -r requirements.txt
streamlit run app.py
```

Or with Docker, from the **repo root** (not from inside `deployment/` — see the note at the top
of `deployment/Dockerfile` about why):

```bash
docker build -f deployment/Dockerfile -t pneumonia-classifier .
docker run -p 8501:8501 pneumonia-classifier
```

Open http://localhost:8501.

## 3. Deploy and run in GitHub Codespaces

1. On GitHub, open your pushed repository.
2. Click **Code > Codespaces > Create codespace on main**.
3. Codespaces reads `.devcontainer/devcontainer.json` at the repo root and builds the same
   `deployment/Dockerfile` image automatically.
4. Once the codespace is running, open a terminal inside it and run:
   ```bash
   streamlit run deployment/app.py --server.port 8501 --server.address 0.0.0.0
   ```
   (If the container's `ENTRYPOINT` already starts the app for you, skip this step and go
   straight to the forwarded port.)
5. Codespaces will show a **"Ports"** tab with port `8501` forwarded — click the globe icon to
   open the forwarded URL, or it opens automatically because `onAutoForward` is set to
   `openPreview` in `devcontainer.json`.
6. Upload a `.dcm` chest X-ray file (e.g. one from `stage_2_test_images/` that was **not** used
   in your model's test split, to keep the demo honest) and confirm you get a class prediction
   and probability back. Screenshot this for the report's Model Deployment section.

## Files in this folder

| File | Purpose |
|---|---|
| `app.py` | Streamlit frontend + inference backend in one file |
| `requirements.txt` | Python dependencies for the app |
| `Dockerfile` | Container image definition (build context = repo root) |
| `model_artifacts/` | Serialized model + config, written by the notebook |

`../.devcontainer/devcontainer.json` (repo root) is what makes step 3 (Codespaces) work
automatically.
