import shutil
import tempfile
from pathlib import Path

import os
import numpy as np
import SimpleITK as sitk
import nibabel as nib


class TemporaryFolder:
    def __init__(self, prefix="temp_folder_"):
        self.prefix = prefix
        self.path = None

    def __enter__(self):
        # создаём временную папку
        self.path = Path(tempfile.mkdtemp(prefix=self.prefix))
        return self.path

    def __exit__(self, exc_type, exc_value, traceback):
        # удаляем папку со всем содержимым
        if self.path and self.path.exists():
            shutil.rmtree(self.path)


def load_arr(input_path: str) -> np.ndarray:
    if os.path.isdir(input_path):
        # --- DICOM ---
        reader = sitk.ImageSeriesReader()
        series_ids = reader.GetGDCMSeriesIDs(input_path)
        if not series_ids:
            raise ValueError(f"No DICOM series found in folder {input_path}")
        series_files = reader.GetGDCMSeriesFileNames(input_path, series_ids[0])
        reader.SetFileNames(series_files)
        sitk_img = reader.Execute()

        # --- схлопываем лишнюю 4-ю размерность, если есть ---
        if sitk_img.GetDimension() == 4:
            # оставляем только первый канал/время
            size = list(sitk_img.GetSize())    # [X, Y, Z, C]
            size[3] = 0                        # схлопываем четвертую размерность
            index = [0, 0, 0, 0]
            extractor = sitk.ExtractImageFilter()
            extractor.SetSize(size)
            extractor.SetIndex(index)
            sitk_img = extractor.Execute(sitk_img)

        # --- приводим к RAS (DICOM по умолчанию LPS -> RAS) ---
        orient_filter = sitk.DICOMOrientImageFilter()
        orient_filter.SetDesiredCoordinateOrientation('RAS')
        sitk_img = orient_filter.Execute(sitk_img)

        # --- конвертируем в numpy (Z,Y,X) -> (X,Y,Z) ---
        arr = sitk.GetArrayFromImage(sitk_img).astype(np.float32)
        arr = np.transpose(arr, (2,1,0))

    else:
        # --- NIfTI ---
        img = nib.load(input_path)
        arr = np.asarray(img.get_fdata(), dtype=np.float32)  # уже (X,Y,Z)
        # при желании можно добавить здесь обработку ориентации для унификации

    return arr
