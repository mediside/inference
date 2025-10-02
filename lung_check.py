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


def solve_lungs(path):
    img_np, meta = LoadImage(image_only=False)(path)
    inferer = LMInferer()
    ans_seg = get_seg_mask(inferer, img_np.numpy())
    ans_meta = lungs_in_meta(meta)
    if ans_seg == 'YES' or ans_meta == 'YES':
        return "YES"
    return "NO"
