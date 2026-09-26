# Pneumonia Detection from Chest X-Rays — Capstone Project

Everything needed to go from your two Google Drive image folders
(`Capstone Project/stage_2_train_images`, `Capstone Project/stage_2_test_images`) to a trained,
compared, deployed model and two submission-ready reports.

## What's in this folder

```
pneumonia_capstone/
├── src/
│   ├── preprocessing.py      # DICOM -> model-ready array (shared by training + deployment)
│   └── data_audit.py         # fixes the train/test patient-overlap + CSV dedup issues
├── notebooks/
│   └── Capstone_Pneumonia_Classification.ipynb   # run this in Google Colab
├── deployment/
│   ├── app.py                # Streamlit inference app
│   ├── requirements.txt
│   ├── Dockerfile
│   ├── model_artifacts/      # populated by the notebook (Section 7)
│   └── README.md             # Docker + GitHub Codespaces walkthrough
├── .devcontainer/
│   └── devcontainer.json     # makes `deployment/Dockerfile` run in GitHub Codespaces
├── reports/
│   ├── report_builder.py     # shared docx formatting helpers
│   ├── generate_interim_report.py
│   ├── generate_final_report.py
│   └── requirements.txt
└── results/                  # created by the notebook: results.json + figures/
```

## The data issue this project fixes

An earlier scoping pass over this dataset found that some patients had an image file in **both**
`stage_2_train_images/` and `stage_2_test_images/` on Google Drive — a data-leakage risk that
would make any test-set accuracy number unreliable. It also found a 543-row gap between the
label CSV and the image count.

`src/data_audit.py` resolves both: it pools every uniquely-labeled patient from both folders,
resolves each overlapping patient to a single file, reconciles labels against what images
actually exist, and then creates a **brand-new, leakage-free, stratified** train/val/test split
with a hard assertion that no patient ID appears in more than one split. The notebook's Section 2
runs this and prints exactly how many overlap/mismatch cases it found in your copy of the data.

## How to run everything, end to end

### 1. Train (Google Colab)

1. Upload this whole `pneumonia_capstone/` folder to your Google Drive, OR just upload
   `notebooks/Capstone_Pneumonia_Classification.ipynb` to Colab — it writes `src/preprocessing.py`
   and `src/data_audit.py` for itself via `%%writefile` cells, so it doesn't need the rest of the
   repo to run.
2. Open the notebook in Colab, enable a GPU runtime (**Runtime > Change runtime type > GPU**).
3. Edit the `PROJECT_ROOT` path in the Setup section if your Drive layout differs from
   `My Drive/Capstone Project/`.
4. Run all cells top to bottom (**Runtime > Run all**). This will:
   - Audit and fix the data leakage/mismatch issues (Section 2)
   - Run EDA and preprocessing (Sections 3-4)
   - Train two from-scratch CNNs, same architecture but different optimizers (Section 5) — this
     alone covers the **Interim submission**
   - Train and compare 3 transfer-learning models, pick a winner, and serialize it (Section 6)
   - Export deployment artifacts (Section 7)
   - Write `results/results.json` + `results/figures/*.png` (Section 9), copied to your Drive too

### 2. Generate the two reports (run locally, or in Colab, wherever you have the `results/` folder)

```bash
cd reports
pip install -r requirements.txt
python generate_interim_report.py --results-dir ../results --out Interim_Report.docx
python generate_final_report.py   --results-dir ../results --out Final_Report.docx
```

`generate_interim_report.py` only needs Sections 1–5 of the notebook to have run.
`generate_final_report.py` needs the full notebook (through Section 9), since it also reports on
transfer learning, model selection, and deployment.

Both scripts insert your real numbers, tables, and saved figures automatically, but leave a small
number of clearly marked `[YOUR INTERPRETATION NEEDED: ...]` placeholders — the report guidelines
are explicit that a good report is "storytelling with data," not just numbers, and that part is
intentionally left for you to write once you've seen your own results.

Convert each `.docx` to PDF for submission (e.g. via Word's "Save as PDF", or
`soffice --headless --convert-to pdf Interim_Report.docx`).

### 3. Deploy (Final submission only)

See `deployment/README.md` for the full Docker + GitHub Codespaces walkthrough. In short: push
`src/`, `deployment/`, and `.devcontainer/` to a GitHub repo, open it in Codespaces, and it builds
and serves the Streamlit app automatically with the port forwarded.

### 4. Submit

Per the report guidelines, each submission needs **two files**:
1. The project report as PDF (`Interim_Report.pdf` / `Final_Report.pdf`)
2. The notebook exported as HTML:
   ```bash
   jupyter nbconvert --to html notebooks/Capstone_Pneumonia_Classification.ipynb
   ```

## Notes on modeling choices

- **Baseline comparison (Interim requirement)**: per mentor guidance, Section 5 builds *two*
  from-scratch CNNs with the identical architecture, changing only the optimizer -- Adam vs. SGD
  with momentum -- so any performance gap is attributable to the optimization procedure, not to
  capacity or regularization differences. The notebook prints a side-by-side comparison table and
  chart, then automatically carries forward whichever scored higher on macro-F1 as "the baseline"
  for the Section 6 transfer-learning comparison; both reports render the two-model comparison
  automatically.
- **Business framing**: model selection is driven by macro-F1 and, as a tiebreaker, recall on
  `Lung Opacity` specifically — a missed pneumonia case is the costliest error for this use case,
  so accuracy alone (which the majority class can inflate) is deliberately not the primary
  criterion. See Section 6.5 of the notebook / final report for the exact rule.
- **Grayscale handling**: chest X-ray DICOMs are natively single-channel; `src/preprocessing.py`
  reads them as grayscale directly (with a fallback that averages down any 3-channel export) and
  replicates to 3 channels only where a pretrained backbone requires it.
- **No train/serve skew**: the Streamlit app imports the exact same `src/preprocessing.py` module
  used during training.
