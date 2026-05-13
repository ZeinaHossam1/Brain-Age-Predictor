import nibabel as nib
import numpy as np
import cv2
import tempfile, os

def load_and_preprocess(file_bytes: bytes, filename: str) -> np.ndarray:
    filename_lower = filename.lower()

    if not (filename_lower.endswith(".nii") or filename_lower.endswith(".nii.gz")):
        raise ValueError("Please upload a .nii or .nii.gz MRI file.")

    suffix = ".nii.gz" if filename_lower.endswith(".nii.gz") else ".nii"

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
        img  = nib.load(tmp_path)
        data = img.get_fdata()
    finally:
        os.unlink(tmp_path)

    print(f"📐 Raw MRI shape: {data.shape}")

    # ── Exact same 5-slice averaging as training ──────────────────
    z      = data.shape[2]
    slice1 = data[:, :, z // 5]
    slice2 = data[:, :, 2 * z // 5]
    slice3 = data[:, :, z // 2]
    slice4 = data[:, :, 3 * z // 5]
    slice5 = data[:, :, 4 * z // 5]
    slice_img = (slice1 + slice2 + slice3 + slice4 + slice5) / 5

    # Normalize to [0, 1]
    if slice_img.max() > 0:
        slice_img = slice_img / slice_img.max()

    # Resize to 128×128
    slice_img = cv2.resize(slice_img, (128, 128))

    # ── Convert to RGB exactly as training ───────────────────────
    # np.concatenate([X * 255, X * 255, X * 255], axis=-1)
    slice_img = slice_img[..., np.newaxis]          # (128, 128, 1)
    rgb = np.concatenate([
        slice_img * 255,
        slice_img * 255,
        slice_img * 255
    ], axis=-1).astype(np.float32)                  # (128, 128, 3)

    # Add batch dim → (1, 128, 128, 3)
    return rgb[np.newaxis, ...]