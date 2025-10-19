import os
import numpy as np
import nibabel as nib
from lungmask import mask
from monai.transforms import LoadImage

from utils import load_arr


def segment_case_sitk(input_path: str, save_dir: str):
    """
    Загружает один случай (nii/nii.gz файл или папку с DICOM),
    применяет lungmask, сохраняет маску как .nii.gz в save_dir.
    Имя выходного файла формируется автоматически: <basename>_mask.nii.gz
    """

    # --- Конвертируем в numpy массив (D,H,W) ---
    arr = load_arr(input_path)
    print(arr.shape)
    # --- Предобработка для lungmask ---
    arr[arr < -1024] = -1024
    arr[arr > 300] = 300

    # --- Сегментация ---
    segmentation = mask.apply(arr)
    print(segmentation.shape)
    # --- Имя для сохранения ---
    base = os.path.basename(input_path.rstrip("/"))
    base = base.replace(".nii.gz", "").replace(".nii", "")
    out_name = f"{base}_mask.nii.gz"
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, out_name)

    # --- Сохраняем маску ---
    seg_nii = nib.Nifti1Image(segmentation.astype(np.uint8), None)
    nib.save(seg_nii, out_path)

    return out_path
