from monai.transforms import LoadImage
import numpy as np

import re
from typing import Any, Dict, List

import os
import numpy as np
import SimpleITK as sitk
from monai.transforms import LoadImage
from lungmask import LMInferer
from monai.data import MetaTensor


def get_img_meta(path):
    img_np, meta = LoadImage(image_only=False)(path)
    return img_np, meta


def _as_numpy(img_obj):
    """Convert MONAI MetaTensor / torch / numpy to numpy array."""
    if isinstance(img_obj, MetaTensor):
        try:
            return img_obj.asnumpy()
        except Exception:
            return np.array(img_obj)
    # torch tensor
    if hasattr(img_obj, "numpy"):
        try:
            return img_obj.numpy()
        except Exception:
            return np.array(img_obj)
    return np.array(img_obj)

def _spacing_direction_from_affine(affine):
    """
    Given a 4x4 affine (nibabel-style), return (spacing_tuple (x,y,z), direction_tuple length 9)
    Best-effort: spacing = norms of affine columns; direction = normalized columns flattened
    into SimpleITK order (row-major: x-axis cosines, then y-axis, then z-axis).
    """
    a = np.array(affine, dtype=float)
    if a.shape != (4, 4):
        raise ValueError("Affine must be 4x4")
    mat = a[:3, :3]  # columns: voxel i,j,k -> world vectors
    spacing_cols = np.linalg.norm(mat, axis=0)
    spacing_cols_safe = np.where(spacing_cols == 0, 1.0, spacing_cols)
    dirs = mat / spacing_cols_safe  # normalized columns
    direction = tuple(float(x) for x in dirs.flatten(order='F'))
    spacing = (float(spacing_cols[0]), float(spacing_cols[1]), float(spacing_cols[2]))
    return spacing, direction

def monai_load_as_sitk(path_or_obj):
    """
    Load image with MONAI LoadImage (to get meta) and return (sitk_image, meta_dict).
    Input can be a path (file or folder for DICOM series) or any object LoadImage accepts.
    """
    loader = LoadImage(image_only=False)
    img_obj, meta = loader(path_or_obj)
    # try to get filename_or_obj (may be string path or list of filenames)
    fn = None
    for k in ("filename_or_obj", "filename", "orig_filename", "file_name"):
        if k in meta:
            fn = meta[k]
            break

    # If filename exists and points to file or folder, prefer to read with SimpleITK (preserve metadata)
    if fn:
        try:
            # Some readers return list/tuple of filenames (for DICOM series). Handle that.
            if isinstance(fn, (list, tuple)) and len(fn) > 0:
                # If it's a list of file paths, use the first path's folder or the list directly
                # ImageSeriesReader will handle stacking if we give the list of files.
                sitk_img = sitk.ReadImage(list(fn))
            elif isinstance(fn, str) and os.path.exists(fn):
                # If it's a file or folder path
                # If it's a folder (DICOM folder), let ImageSeriesReader handle it:
                if os.path.isdir(fn):
                    # find series IDs and read first series
                    reader = sitk.ImageSeriesReader()
                    series_IDs = reader.GetGDCMSeriesIDs(fn)
                    if series_IDs:
                        series_file_names = reader.GetGDCMSeriesFileNames(fn, series_IDs[0])
                        reader.SetFileNames(series_file_names)
                        sitk_img = reader.Execute()
                    else:
                        # maybe folder contains single .nii etc.
                        # try reading the folder as a single file path (will fail usually)
                        raise RuntimeError("No DICOM series found in folder; falling back to array conversion.")
                else:
                    # regular file path (.nii/.nii.gz/.mhd/.nrrd etc.)
                    sitk_img = sitk.ReadImage(fn)
            else:
                # fn exists but isn't a filesystem path we can use; fallback to array conversion
                sitk_img = None
            if sitk_img is not None:
                return sitk_img, meta
        except Exception:
            # on any error, we will fallback to array conversion below
            pass

    # Fallback: convert loaded array -> SimpleITK
    arr = _as_numpy(img_obj)  # expected shape: (C,H,W) or (D,H,W) or (H,W)
    # MONAI often returns channel-first arrays (C,H,W[,D]) — try to put into (D,H,W)
    # Heuristics:
    if arr.ndim == 4 and arr.shape[0] == 1:
        # (1, D, H, W) or (1, H, W, D) uncommon; assume (C,D,H,W) -> drop channel
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[0] <= 4 and arr.shape[0] != arr.shape[1] and arr.shape[0] != arr.shape[2]:
        # ambiguous: could be (C,D,H) or (D,H,W). If first dim is small (<=4) and equals channels, assume channel-first
        # But safest is: if meta contains "num_channels" or "scalars_per_pixel" use it. For simplicity, we assume arr is (D,H,W)
        pass

    # If arr is (H,W) -> GetImageFromArray will make 2D image
    sitk_img = sitk.GetImageFromArray(arr)  # note: this interprets arr as (z,y,x)

    # Try to set spacing/direction/origin from meta if available (best-effort)
    # 1) affine (nibabel) -> compute spacing+direction
    if "affine" in meta and meta["affine"] is not None:
        try:
            spacing, direction = _spacing_direction_from_affine(meta["affine"])
            sitk_img.SetSpacing(spacing)
            sitk_img.SetDirection(direction)
        except Exception:
            pass

    # 2) other common spacing keys
    for key in ("spacing", "pixdim", "original_spacing", "spacing_mm"):
        if key in meta and meta[key] is not None:
            try:
                sp = meta[key]
                sp = tuple(float(x) for x in (sp if len(sp) >= 3 else (sp[0], sp[1], sp[2])))
                # Note: MONAI spacing order might be (z,y,x) or (x,y,z); we try to detect common cases:
                if len(sp) == 3:
                    # sitk expects (x,y,z)
                    # if meta uses (z,y,x) (nibabel sometimes stores pixdim as [x,y,z] though),
                    # we attempt to detect by comparing arr shape and spacing: skip detection complexity for brevity
                    sitk_img.SetSpacing((sp[2], sp[1], sp[0]))
            except Exception:
                pass
            break

    # 3) origin / direction from meta if present (best-effort)
    if "direction" in meta and meta["direction"] is not None:
        try:
            dir_meta = meta["direction"]
            dir_flat = tuple(float(x) for x in np.array(dir_meta).flatten())
            if len(dir_flat) == 9:
                sitk_img.SetDirection(dir_flat)
        except Exception:
            pass

    if "origin" in meta and meta["origin"] is not None:
        try:
            origin = tuple(float(x) for x in meta["origin"])
            sitk_img.SetOrigin(origin)
        except Exception:
            pass

    return sitk_img, meta

def get_seg_mask(inferer: LMInferer, sitk_img) -> str:
    """
    Запускает inferer.apply на sitk_img и возвращает "YES" если в маске есть ненулевые воксели, иначе "NO".
    Поддерживает случаи, когда inferer.apply возвращает SimpleITK.Image ИЛИ numpy.ndarray.
    """
    seg = inferer.apply(sitk_img)

    # Если вернулся SimpleITK.Image
    if isinstance(seg, sitk.SimpleITK.Image):
        arr = sitk.GetArrayFromImage(seg)
    # Если вернулся numpy array
    elif isinstance(seg, np.ndarray):
        arr = seg
    # Если вернулся PyTorch tensor (маловероятно, но на всякий случай)
    else:
        try:
            import torch
            if isinstance(seg, torch.Tensor):
                arr = seg.detach().cpu().numpy()
            else:
                # Попытка привести к numpy
                arr = np.array(seg)
        except Exception:
            # последний запасной вариант — ошибка
            raise TypeError(f"Unsupported segmentation return type: {type(seg)}")

    # проверка ненулевых вокселей (при float используем > 0.5 если нужно)
    if (arr > 0).any():
        return "YES"
    else:
        return "NO"
    

def _gather_strings(x: Any) -> List[str]:
    """Рекурсивно собрать все строковые представления из объекта meta."""
    out = []
    if x is None:
        return out
    if isinstance(x, str):
        out.append(x)
    elif isinstance(x, (list, tuple, set)):
        for it in x:
            out.extend(_gather_strings(it))
    elif isinstance(x, dict):
        for v in x.values():
            out.extend(_gather_strings(v))
    else:
        # попытка привести к строке (например числа, пути и т.д.)
        try:
            out.append(str(x))
        except Exception:
            pass
    return out

# список позитивных и негативных ключевых слов (маленькими буквами)
_POSITIVE_KEYWORDS = [
    "lung", "lungs", "chest", "thorax", "thoracic", "pulmonary", "pulmo",
    "pleura", "bronch", "airway", "ct_chest", "ct chest", "chest ct", "cta chest",
    "lung_window", "lung window", "pulmonary embol", "pulm", "respiratory"
]

_NEGATIVE_KEYWORDS = [
    "brain", "head", "abdomen", "pelvis", "knee", "spine", "dental", "neck",
    "shoulder", "wrist", "elbow", "mri", "ultrasound", "us", "mammography", "breast"
]

def lungs_in_meta(meta: Dict) -> str:
    """
    Принимает `meta` (image_meta_dict из LoadImage(..., image_only=False))
    Возвращает "YES" если по метаданным вероятно, что это обследование грудной клетки/лёгких,
    иначе "NO".
    """
    if not isinstance(meta, dict):
        raise ValueError("meta must be a dict (image_meta_dict from MONAI LoadImage).")

    # собрать все строковые фрагменты из meta
    strings = _gather_strings(meta)
    if not strings:
        return "NO"

    # объединить и нормализовать текст
    combined = " ".join(s.lower() for s in strings if isinstance(s, str))
    # заменить разделители на пробелы
    combined = re.sub(r"[_\-/\\\.\,\:]", " ", combined)

    # подсчёт вхождений по целым словам (чтобы, например, 'lung' не совпадал в середине длинных слов)
    def count_keywords(keywords):
        total = 0
        for kw in keywords:
            pattern = r"\b" + re.escape(kw.lower()) + r"\b"
            total += len(re.findall(pattern, combined))
        return total

    pos_count = count_keywords(_POSITIVE_KEYWORDS)
    neg_count = count_keywords(_NEGATIVE_KEYWORDS)

    if pos_count > 0 and pos_count >= neg_count:
        return "YES"

    dicom_keys = ["body_part_examined", "series_description", "protocol_name", "study_description", "series_description"]
    for k in dicom_keys:
        if k in meta:
            val = meta[k]
            # если значение строка, посмотрим в ней отдельно
            if isinstance(val, str):
                v = val.lower()
                for kw in _POSITIVE_KEYWORDS:
                    if re.search(r"\b" + re.escape(kw) + r"\b", v):
                        return "YES"

    return "NO"


def solve_lungs(volume, meta):
    # img_np, meta = monai_load_as_sitk(path)
    inferer = LMInferer()
    ans_seg = get_seg_mask(inferer, volume)
    ans_meta = lungs_in_meta(meta)
    if ans_seg == 'YES' or ans_meta == 'YES':
        return "YES"
    return "NO"
