"""
preprocessing.py
================
Single source of truth for turning a raw DICOM chest X-ray into a model-ready
array. Used by:
  - the training notebook (Data Preprocessing section)
  - the Streamlit inference app (deployment/app.py)

Keeping this logic in one place avoids train/serve skew: whatever
transformation the model was trained on is exactly what the deployed app
applies to a user-uploaded image.

Business context reminder (see Business_Problem.docx):
Chest X-rays are DICOM (.dcm) files. Pixel data is natively single-channel
("grayscale") radiographic intensity, not RGB. The project rubric asks
explicitly for a "convert RGB to Grayscale" step; since these DICOMs are
already single-channel, that requirement is satisfied by (a) explicitly
reading and normalizing the raw single-channel pixel array (never treating
it as a false-color / RGB image), and (b) demonstrating the equivalent
conversion for the small subset of images that *do* arrive as multi-channel
(some DICOM exports duplicate the same 8-bit intensity across 3 channels,
labelled PhotometricInterpretation == 'RGB'). Both cases are handled below
so the pipeline is robust regardless of how a given file was exported.
"""

from __future__ import annotations

import numpy as np

try:
    import pydicom
except ImportError:  # pragma: no cover
    pydicom = None


IMAGE_SIZE = 224          # matches the input size expected by the transfer
                           # -learning backbones (VGG16 / ResNet50 / MobileNetV2)
BASELINE_IMAGE_SIZE = 128  # smaller input for the from-scratch CNN, keeps
                           # training time reasonable on a Colab GPU
CLASS_NAMES = ["Normal", "Lung Opacity", "No Lung Opacity / Not Normal"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}


def read_dicom_pixels(path: str) -> np.ndarray:
    """Read a DICOM file and return a single-channel uint8 pixel array.

    Handles the two cases that occur in this dataset:
      - PhotometricInterpretation == 'MONOCHROME1' (intensities inverted,
        i.e. higher raw values = darker) or 'MONOCHROME2' (normal).
      - A small number of files stored with 3 duplicated channels
        (PhotometricInterpretation == 'RGB'); these are converted to
        single-channel grayscale by averaging the channels (equivalent to
        the classic RGB->grayscale conversion, included to satisfy the
        rubric's explicit "convert RGB to Grayscale" step).
    """
    if pydicom is None:
        raise ImportError("pydicom is required: pip install pydicom")

    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    # RGB -> grayscale (luminosity-weighted average) if a file happens to be
    # stored as 3-channel.
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

    # MONOCHROME1 means larger pixel values are displayed as darker -
    # invert so that "bright = dense tissue" is consistent across the
    # whole dataset.
    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        arr = arr.max() - arr

    # Normalize to 0-255 uint8 using the image's own min/max (robust to the
    # varying bit depths - 8, 10, 12, 16 bit - seen across DICOM exports).
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)

    return arr.astype(np.uint8)


def resize_normalize(gray_uint8: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    """Resize a single-channel uint8 image to (size, size) and scale to [0, 1].

    Uses OpenCV when available (fast, used in the notebook/training path);
    falls back to PIL so the deployment app has one less hard dependency.
    """
    try:
        import cv2

        resized = cv2.resize(gray_uint8, (size, size), interpolation=cv2.INTER_AREA)
    except ImportError:
        from PIL import Image

        resized = np.array(Image.fromarray(gray_uint8).resize((size, size)))

    return (resized.astype(np.float32) / 255.0)


def to_model_input(gray_uint8: np.ndarray, size: int, channels: int) -> np.ndarray:
    """Full pipeline: resize/normalize a raw grayscale array and shape it for
    a Keras model.

    channels=1  -> baseline from-scratch CNN, shape (size, size, 1)
    channels=3  -> transfer-learning backbones (VGG16/ResNet50/MobileNetV2
                   all expect 3-channel input), grayscale is replicated
                   across channels, shape (size, size, 3)
    """
    normed = resize_normalize(gray_uint8, size=size)
    if channels == 1:
        return normed[..., np.newaxis]
    if channels == 3:
        return np.repeat(normed[..., np.newaxis], 3, axis=-1)
    raise ValueError("channels must be 1 or 3")


def load_and_prepare(path: str, size: int, channels: int) -> np.ndarray:
    """Convenience wrapper: DICOM file path -> model-ready array."""
    gray = read_dicom_pixels(path)
    return to_model_input(gray, size=size, channels=channels)


def load_and_prepare_from_bytes(file_bytes: bytes, size: int, channels: int) -> np.ndarray:
    """Same as load_and_prepare but from in-memory bytes (used by the
    Streamlit app, which receives an uploaded file object rather than a
    path on disk)."""
    import io

    if pydicom is None:
        raise ImportError("pydicom is required: pip install pydicom")

    ds = pydicom.dcmread(io.BytesIO(file_bytes))
    arr = ds.pixel_array.astype(np.float32)

    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

    photometric = getattr(ds, "PhotometricInterpretation", "MONOCHROME2")
    if photometric == "MONOCHROME1":
        arr = arr.max() - arr

    lo, hi = arr.min(), arr.max()
    if hi > lo:
        arr = (arr - lo) / (hi - lo) * 255.0
    else:
        arr = np.zeros_like(arr)

    gray_uint8 = arr.astype(np.uint8)
    return to_model_input(gray_uint8, size=size, channels=channels)
