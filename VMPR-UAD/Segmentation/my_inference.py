import os
import numpy as np
import nibabel as nib
from lungmask import mask
from monai.transforms import LoadImage


def segment_case(input_path: str, save_dir: str):
    """
    Загружает один случай (nii/nii.gz файл или папку с DICOM),
    применяет lungmask, сохраняет маску как .nii.gz в save_dir.
    Имя выходного файла формируется автоматически: <basename>_mask.nii.gz
    """
    loader = LoadImage(image_only=True)

    # загрузка изображения (nii/dicom)
    img = loader(input_path)
    arr = np.asarray(img)
    print(arr.shape)
    # intensity clipping (как в оригинале)
    arr[arr < -1024] = -1024

    # сегментация
    segmentation = mask.apply(arr)
    print(segmentation.shape)
    # имя для сохранения
    base = os.path.basename(input_path.rstrip("/"))  # убираем слэш для папок
    if base.endswith(".nii.gz"):
        base = base.replace(".nii.gz", "")
    elif base.endswith(".nii"):
        base = base.replace(".nii", "")
    out_name = f"{base}_mask.nii.gz"

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, out_name)

    # сохраняем маску
    seg_nii = nib.Nifti1Image(segmentation.astype(np.uint8), None)
    nib.save(seg_nii, out_path)

    return out_path

import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from lungmask import mask


# def load_arr(input_path):
#     # --- Загружаем изображение через SimpleITK ---
#     if os.path.isdir(input_path):
#         # предполагаем DICOM серию
#         reader = sitk.ImageSeriesReader()
#         series_ids = reader.GetGDCMSeriesIDs(input_path)
#         if not series_ids:
#             raise ValueError(f"No DICOM series found in folder {input_path}")
#         series_files = reader.GetGDCMSeriesFileNames(input_path, series_ids[0])
#         reader.SetFileNames(series_files)
#         sitk_img = reader.Execute()
#     else:
#         # файл NIfTI или одиночный DICOM
#         sitk_img = sitk.ReadImage(input_path)

#     # --- Конвертируем в numpy массив (D,H,W) ---
#     arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32).squeeze()
#     return np.transpose(arr, (2, 1, 0))   
import SimpleITK as sitk
import nibabel as nib
import os
import numpy as np

def load_arr(input_path: str) -> np.ndarray:
    if os.path.isdir(input_path):
        # DICOM
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(input_path)
        if not series_ids:
            raise ValueError(f"No DICOM series found in folder {input_path}")
        series_files = reader.GetGDCMSeriesFileNames(input_path, series_ids[0])
        reader.SetFileNames(series_files)
        sitk_img = reader.Execute()
        orient_filter = sitk.DICOMOrientImageFilter()
        orient_filter.SetDesiredCoordinateOrientation('RAS')
        sitk_img = orient_filter.Execute(sitk_img)
        arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        arr = np.transpose(arr, (2,1,0))  # (X,Y,Z)
    else:
        # NIfTI
        img = nib.load(input_path)
        arr = np.asarray(img.get_fdata(), dtype=np.float32)  # уже (X,Y,Z)
    return arr


def segment_case_sitk(input_path: str, save_dir: str):
    """
    Загружает один случай (nii/nii.gz файл или папку с DICOM),
    применяет lungmask, сохраняет маску как .nii.gz в save_dir.
    Имя выходного файла формируется автоматически: <basename>_mask.nii.gz
    """
    # --- Загружаем изображение через SimpleITK ---
    # if os.path.isdir(input_path):
    #     # предполагаем DICOM серию
    #     reader = sitk.ImageSeriesReader()
    #     series_ids = reader.GetGDCMSeriesIDs(input_path)
    #     if not series_ids:
    #         raise ValueError(f"No DICOM series found in folder {input_path}")
    #     series_files = reader.GetGDCMSeriesFileNames(input_path, series_ids[0])
    #     reader.SetFileNames(series_files)
    #     sitk_img = reader.Execute()
    # else:
    #     # файл NIfTI или одиночный DICOM
    #     sitk_img = sitk.ReadImage(input_path)

    # --- Конвертируем в numpy массив (D,H,W) ---
    arr = load_arr(input_path) #sitk.GetArrayFromImage(sitk_img).astype(np.float32)
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
