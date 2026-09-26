"""
data_audit.py
=============
Fixes the two data-quality issues flagged in the earlier project scoping
conversation (see CNN_Lung_Classification_Project_Summary.md):

  1. Patient overlap between the `stage_2_train_images` and
     `stage_2_test_images` Google Drive folders -> data leakage risk.
  2. A row-count mismatch between `stage_2_detailed_class_info.csv`
     (30,227 rows) and the number of physical images (29,684) -> caused by
     the CSV carrying one row per bounding box for Target=1 (Lung Opacity)
     patients, not one row per image. A patient never has more than one
     image file (files are named `{patientId}.dcm`, one per patient), so
     the fix is a straightforward de-duplication, not a multi-image merge.

Design decision on the leakage fix
-----------------------------------
Rather than patch the existing train/test folder assignment (which is
already compromised - we cannot tell, after the fact, which of the two
copies of an overlapping patient the original split "intended" to use),
this module POOLS every uniquely-labeled patient found in either folder
and creates a brand-new stratified train/validation/test split at the
patient level. Because each patient contributes exactly one image, a
patient-level split is automatically leakage-free (no group-based
splitting is required beyond simply not letting a patientId appear in more
than one split, which `make_clean_split` guarantees).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass
class AuditReport:
    n_csv_rows_raw: int
    n_unique_patients_csv: int
    n_train_folder_files: int
    n_test_folder_files: int
    n_overlap_patients: int
    overlap_examples: list
    n_labeled_no_image: int
    n_image_no_label: int
    n_final_usable_patients: int
    class_distribution: dict
    inconsistent_label_patients: list

    def to_dict(self) -> dict:
        return self.__dict__


def _ids_from_folder(folder: str) -> set:
    """Return the set of patientIds (filename stem) present in a folder of
    .dcm files."""
    return {
        os.path.splitext(f)[0]
        for f in os.listdir(folder)
        if f.lower().endswith(".dcm")
    }


def audit_and_build_master_table(
    detailed_class_csv: str,
    train_images_folder: str,
    test_images_folder: str,
) -> tuple[pd.DataFrame, AuditReport]:
    """Run the full data-quality audit and return a clean, leakage-free
    master table plus a report of every issue found (for the Data
    Overview / EDA sections of the report).

    Returns
    -------
    master_df : DataFrame with columns
        ['patientId', 'class', 'file_path', 'source_folder']
        - one row per patient that has BOTH a label and a resolvable image.
    report : AuditReport
        Every number needed to write the "Data Quality Issues Found /
        Actions Taken" narrative in the Interim and Final reports.
    """
    raw = pd.read_csv(detailed_class_csv)
    n_csv_rows_raw = len(raw)

    # --- Issue 2: de-duplicate the CSV to one row per patient -------------
    # Sanity check first: does every duplicated patientId agree on `class`?
    dup_check = raw.groupby("patientId")["class"].nunique()
    inconsistent_label_patients = dup_check[dup_check > 1].index.tolist()

    labels = raw.drop_duplicates(subset=["patientId"], keep="first").copy()
    n_unique_patients_csv = len(labels)

    # --- Issue 1: find the train/test patient overlap ----------------------
    train_ids = _ids_from_folder(train_images_folder)
    test_ids = _ids_from_folder(test_images_folder)
    overlap = train_ids & test_ids

    # --- Resolve overlap + missing files -----------------------------------
    # Policy: an overlapping patient's single physical file is kept once,
    # preferring the copy in the train folder (arbitrary but documented and
    # consistent - it does not matter which physical copy we keep since the
    # file content for a given patientId is identical in both folders; what
    # matters is that the patient is only usable by ONE split downstream).
    resolved_rows = []
    for pid in sorted(train_ids | test_ids):
        if pid in train_ids:
            resolved_rows.append((pid, os.path.join(train_images_folder, f"{pid}.dcm"), "train_folder"))
        else:
            resolved_rows.append((pid, os.path.join(test_images_folder, f"{pid}.dcm"), "test_folder"))
    files_df = pd.DataFrame(resolved_rows, columns=["patientId", "file_path", "source_folder"])

    labeled_ids = set(labels["patientId"])
    all_image_ids = train_ids | test_ids
    n_labeled_no_image = len(labeled_ids - all_image_ids)
    n_image_no_label = len(all_image_ids - labeled_ids)

    master_df = files_df.merge(labels[["patientId", "class"]], on="patientId", how="inner")
    master_df = master_df[["patientId", "class", "file_path", "source_folder"]].reset_index(drop=True)

    report = AuditReport(
        n_csv_rows_raw=n_csv_rows_raw,
        n_unique_patients_csv=n_unique_patients_csv,
        n_train_folder_files=len(train_ids),
        n_test_folder_files=len(test_ids),
        n_overlap_patients=len(overlap),
        overlap_examples=sorted(overlap)[:5],
        n_labeled_no_image=n_labeled_no_image,
        n_image_no_label=n_image_no_label,
        n_final_usable_patients=len(master_df),
        class_distribution=master_df["class"].value_counts().to_dict(),
        inconsistent_label_patients=inconsistent_label_patients[:5],
    )
    return master_df, report


def make_clean_split(
    master_df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    random_state: int = 42,
) -> pd.DataFrame:
    """Create a brand-new, leakage-free, stratified train/val/test split.

    Every patientId appears in exactly one split (guaranteed trivially -
    the source table already has one row per patient), and class
    proportions are preserved in each split via stratification.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-9, "fractions must sum to 1.0"

    df = master_df.copy()
    train_df, temp_df = train_test_split(
        df,
        train_size=train_frac,
        stratify=df["class"],
        random_state=random_state,
    )
    relative_val = val_frac / (val_frac + test_frac)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=relative_val,
        stratify=temp_df["class"],
        random_state=random_state,
    )

    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")

    out = pd.concat([train_df, val_df, test_df], ignore_index=True)

    # Hard assertion: no patient leakage across splits (this is the whole
    # point of the exercise - fail loudly if it's ever violated).
    counts = out.groupby("patientId")["split"].nunique()
    assert (counts == 1).all(), "Leakage detected: a patientId appears in more than one split!"

    return out
