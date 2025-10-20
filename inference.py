import argparse
import os 
import pydicom
import io, zipfile
import torch

import shutil

import cv2

import numpy as np

import matplotlib.pyplot as plt

from totalsegmentator.python_api import totalsegmentator
import nibabel as nib

from utils import TemporaryFolder

import dicom2nifti
from nilearn.image import resample_img

import importlib

test_AD_each_view = importlib.import_module("VMPR-UAD.Anomaly_detection.test_AD_each_view")

DEVICE = 'cuda'
MODEL_PATH = 'model.pt'

MASK_DIRECTORY = 'mask_directory'
PROJECTIONS_DIRECTORY = 'projections_directory'
RESULTS_DIRECTORY = 'results'

NIFTI_FOLDER = "nifti_folder"
# MASK_PATH = "mask_directory_2"

STEP_START = 'start' # скрипт жив и начал инференс
STEP_FILE_READ = 'file_read' # скрипт прочитал файл
STEP_LUNG_CHECK = 'lung_check'
STEP_PREPROCESSING = 'preprocessing' # скрипт закончил препроцессинг
STEP_SEGMENTATION = 'segmentation'
STEP_PROJECTIONS = 'projections'
STEP_INFERENCE = 'inference'
STEP_FINISH = 'finish' # скрипт закончил инференс


def make_projections(image, mask, filename, savepath):
    projection_name = ['r_a', 'r_c', 'r_s', 'l_a', 'l_c', 'l_s']

    dataarray = image
    dataarray[dataarray < -1024] = -1024
    dataarray = normalize(dataarray)

    maskdata = mask 

    kernel = np.ones((3, 3))
    for idx in range(maskdata.shape[0]):
        for _ in range(2):
            try:
                maskdata[idx, :, :] = cv2.erode(maskdata[idx, :, :], kernel, iterations=1)
            except:
                pass

    leftmask = np.zeros(maskdata.shape)
    leftmask[maskdata == 2] = 1
    rightmask = np.zeros(maskdata.shape)
    rightmask[maskdata == 1] = 1
    right = dataarray.copy()  # * rightmask
    left = dataarray.copy()  # * leftmask
    right[rightmask != 1] = right.min()
    left[leftmask != 1] = left.min()
    projection_list=[]

    for idx in range(3):
        projection_list.append(np.max(right.copy(), axis=idx)[::-1, :])

    for idx in range(3):
        projection_list.append(np.max(left.copy(), axis=idx)[::-1, :])


    rightmask = np.array(rightmask, dtype='uint8')
    leftmask = np.array(leftmask, dtype='uint8')
    projection_masklist = []
    for idx in range(3):
        mask=np.max(rightmask.copy(), axis=idx)[::-1, :]
        mask[mask>0]=1
        projection_masklist.append(mask)

    for idx in range(3):
        mask = np.max(leftmask.copy(), axis=idx)[::-1, :]
        mask[mask>0]=1
        projection_masklist.append(mask)

    # patient_name = name.split('/')[-1].replace('.nii.gz', '')

    for position, img, mask in zip(projection_name, projection_list, projection_masklist):
        img, mask = crop_area(img, mask)
        
        # Формируем корректные пути
        img_path = os.path.join(savepath, f"{filename}_{position}.png")
        mask_path = os.path.join(savepath, f"{filename}_{position}_mask.png")
        
        # Сохраняем
        plt.imsave(img_path, img, cmap='gray')
        plt.imsave(mask_path, mask, cmap='gray')



import os
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
import nibabel as nib

LABEL_MAPPING = {
    'scapula_left': 71,
    'scapula_right': 72,
    'clavicula_left': 73,
    'clavicula_right': 74,
    'rib_left_1': 92,
    'rib_left_6': 97,
    'rib_right_1': 104,
    'rib_right_6': 109,
    'sternum': 116,
    'costal_cartilages': 117,
}

BONE_CLASSES = [
    'clavicula_left', 'clavicula_right',
    'scapula_left', 'scapula_right',
    'rib_left_1', 'rib_left_6',
    'rib_right_1', 'rib_right_6',
    'sternum', 'costal_cartilages'
]

import pandas as pd

class_map = pd.Series({
    10:  'lung_upper_lobe_left',
    11:  'lung_lower_lobe_left',
    12:  'lung_upper_lobe_right',
    13:  'lung_middle_lobe_right',
    14:  'lung_lower_lobe_right',
    15:  'esophagus',
    16:  'trachea',
    51:  'heart',
    52:  'aorta',
    53:  'pulmonary_vein',
    54:  'brachiocephalic_trunk',
    55:  'subclavian_artery_right',
    56:  'subclavian_artery_left',
    57:  'common_carotid_artery_right',
    58:  'common_carotid_artery_left',
    59:  'brachiocephalic_vein_left',
    60:  'brachiocephalic_vein_right',
    62:  'superior_vena_cava',
    71:  'scapula_left',
    72:  'scapula_right',
    73:  'clavicula_left',
    74:  'clavicula_right',
    79:  'spinal_cord',
    92:  'rib_left_1',
    97:  'rib_left_6',
    104: 'rib_right_1',
    109: 'rib_right_6',
    116: 'sternum',
    117: 'costal_cartilages',
}, name='TotalSegmentator name')

def make_bones_projections(image_hu, mask, class_map, filename, savepath,
                                  system_names=None,
                                  window_center=300, window_width=1000,
                                  spacing_mm=1.0,
                                  smoothing_sigma=0.6,
                                  pad_px=8):
    """
    Сохраняет ровно 3 PNG:
      {filename}_vessel_axial.png
      {filename}_vessel_coronal.png
      {filename}_vessel_sagittal.png

    - image_hu: 3D (z,y,x) HU
    - mask: 3D маска (id классов)
    - class_map: {id: 'name'}
    - system_names: list имен, которые относятся к 'vessel' (по умолчанию берем все имена из class_map)
    """
    os.makedirs(savepath, exist_ok=True)

    if system_names is None:
        system_names = list(class_map.values)  # <- важно скобки
    system_mask3 = np.zeros_like(mask, dtype=np.uint8)
    for cls_id, cls_name in class_map.items():
        if cls_name in system_names:
            system_mask3[mask == cls_id] = 1

    use_mask = system_mask3.copy()
    mask_is_empty = (use_mask.sum() == 0)
    if mask_is_empty:
        print("Warning: mask is empty -> producing full-volume projections.")

    # --- smoothing + window normalization --------------------------------
    if smoothing_sigma is not None and smoothing_sigma > 0:
        image_smooth = ndimage.gaussian_filter(image_hu, sigma=smoothing_sigma)
    else:
        image_smooth = image_hu.copy()

    lo = window_center - window_width/2.0
    hi = window_center + window_width/2.0
    img_norm = (np.clip(image_smooth, lo, hi) - lo) / (hi - lo + 1e-8)
    img_norm = np.clip(img_norm, 0.0, 1.0)

    # --- helper crop (взято из вашего кода) -------------------------------
    def crop_to_mask_with_padding_local(img2d, mask2d, pad=8):
        if mask2d.sum() == 0:
            return img2d, mask2d
        ys, xs = np.where(mask2d > 0)
        ymin, ymax = ys.min(), ys.max()
        xmin, xmax = xs.min(), xs.max()
        ymin = max(0, ymin - pad)
        xmin = max(0, xmin - pad)
        ymax = min(img2d.shape[0]-1, ymax + pad)
        xmax = min(img2d.shape[1]-1, xmax + pad)
        return img2d[ymin:ymax+1, xmin:xmax+1], mask2d[ymin:ymax+1, xmin:xmax+1]

    axes = {'axial':0, 'coronal':1, 'sagittal':2}
    saved_paths = []

    for view_name, axis in axes.items():
        # полный MIP по выбранному axis
        proj = np.max(img_norm, axis=axis)
        # маска проекции (если маска пустая -> это массив нулей, crop вернёт весь proj)
        mask_proj = np.max(use_mask, axis=axis).astype(np.uint8)

        # чтобы совпадало с предыдущим отображением
        proj = proj[::-1, :]
        mask_proj = mask_proj[::-1, :]

        # crop (если маска пустая — вернётся исходный proj)
        proj_crop, mask_crop = crop_to_mask_with_padding_local(proj, mask_proj, pad=pad_px)

        # если crop пустой (маловероятно) — используем непотретённый proj
        if proj_crop.size == 0:
            proj_crop = proj

        # сохранить как uint8 grayscale
        out_uint8 = (np.clip(proj_crop, 0.0, 1.0) * 255.0).astype(np.uint8)
        out_name = f"{filename}_bones_{view_name}.png"
        out_path = os.path.join(savepath, out_name)
        plt.imsave(out_path, out_uint8, cmap='gray', vmin=0, vmax=255)
        saved_paths.append(out_path)

    print("Saved 3 bones projections:", saved_paths)
    return saved_paths


def crop_area(img,mask):
    # l_idx = 0
    # r_idx = mask.shape[1] - 1
    for idx in range(0, mask.shape[1]):
        if mask[:, idx].sum() != 0:
            l_idx = idx
            break
    for idx in range(mask.shape[1] - 1, 0, -1):
        if mask[:, idx].sum() != 0:
            r_idx = idx
            break
    img = img[:, l_idx:r_idx + 1, ]
    mask = mask[:, l_idx:r_idx + 1]

    # l_idx = 0
    # r_idx = mask.shape[0] - 1
    for idx in range(0, mask.shape[0]):
        if mask[idx, :].sum() != 0:
            l_idx = idx
            break
    for idx in range(mask.shape[0] - 1, 0, -1):
        if mask[idx, :].sum() != 0:
            r_idx = idx
            break
    img = img[l_idx:r_idx + 1, :]
    mask = mask[l_idx:r_idx + 1, :]
    return img,mask

def normalize(volume, max=0,min=-800):
    volume[volume < min] = min
    volume[volume > max] = max
    volume=(volume-min)/(max-min)

    return volume


def extract_dicom_series(zip_path, study_id, series_id, out_dir):
    """
    Извлекает DICOM-файлы из архива, фильтруя по StudyInstanceUID и SeriesInstanceUID,
    и сохраняет их в папку out_dir. Возвращает путь к этой папке.
    """

    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            try:
                # читаем только заголовок
                ds = pydicom.dcmread(io.BytesIO(zf.read(name)), stop_before_pixels=True, force=True)
                if (getattr(ds, "StudyInstanceUID", None) == study_id and
                    getattr(ds, "SeriesInstanceUID", None) == series_id):
                    # сохраняем файл
                    out_path = out_dir / os.path.basename(name)
                    with open(out_path, "wb") as f:
                        f.write(zf.read(name))
            except Exception:
                continue

    return str(out_dir)  # MONAI ждёт путь к папке


def clean_directory(dir_path):
    """Очищает одну директорию"""
    if os.path.exists(dir_path):
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f"Ошибка при удалении {file_path}: {e}")

def setup_environment():
    """Подготавливает окружение: создает и очищает директории"""
    directories = [MASK_DIRECTORY, PROJECTIONS_DIRECTORY, RESULTS_DIRECTORY, NIFTI_FOLDER]
    
    for dir_path in directories:
        # Создаем директорию если не существует
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        # Очищаем директорию
        clean_directory(dir_path)
        print(f"Подготовлена директория: {dir_path}")

def cleanup_environment():
    """Очищает все рабочие директории"""
    directories = [MASK_DIRECTORY, PROJECTIONS_DIRECTORY, RESULTS_DIRECTORY, NIFTI_FOLDER]
    
    for dir_path in directories:
        clean_directory(dir_path)
        print(f"Очищена директория: {dir_path}")


def doInference(file_path: str, study_id: str, series_id: str):

    # Подготовка перед запуском
    setup_environment()

    try:
        print(f'filepath: {file_path}, study_id: {study_id}, series_id: {series_id}')
        yield 0, STEP_START

        yield 10, STEP_FILE_READ

        with TemporaryFolder(prefix="dicom_") as temp_dir:
            dicom_dir = extract_dicom_series(file_path, study_id, series_id, temp_dir)

            yield 20, STEP_LUNG_CHECK
            name = os.path.basename(os.path.normpath(dicom_dir))

            nifti_path = os.path.join(NIFTI_FOLDER, f"{name}.nii.gz")
            mask_dir = os.path.join(MASK_DIRECTORY, name)

            yield 30, STEP_PREPROCESSING

            dicom2nifti.dicom_series_to_nifti(
                dicom_dir,
                nifti_path,
                reorient_nifti=True
            )
            
            input_nifti = nib.load(nifti_path) #.get_fdata()
            target_affine = np.eye(3)  # 1мм изотропный воксель
            input_nifti = resample_img(input_nifti, target_affine=target_affine, interpolation='continuous', copy_header=True)

            yield 40, STEP_SEGMENTATION

            totalsegmentator(
                input_nifti,  
                mask_dir,
                device='gpu',
                fast=False,
                roi_subset=[
                    'lung_upper_lobe_left', 'lung_lower_lobe_left',
                    'lung_upper_lobe_right', 'lung_middle_lobe_right',
                    'lung_lower_lobe_right'
                ] + ['clavicula_left', 'clavicula_right',
                     'scapula_left', 'scapula_right',
                     'rib_left_1', 'rib_left_6',
                     'rib_right_1', 'rib_right_6',
                     'sternum', 'costal_cartilages'],
                nr_thr_saving=1,
            )

            yield 80, STEP_PROJECTIONS

            print("Формируем объединённую маску...")
            mask_upper_right = nib.load(os.path.join(mask_dir, 'lung_upper_lobe_right.nii.gz')).get_fdata()
            mask_middle_right = nib.load(os.path.join(mask_dir, 'lung_middle_lobe_right.nii.gz')).get_fdata()
            mask_lower_right = nib.load(os.path.join(mask_dir, 'lung_lower_lobe_right.nii.gz')).get_fdata()
            mask_upper_left = nib.load(os.path.join(mask_dir, 'lung_upper_lobe_left.nii.gz')).get_fdata()
            mask_lower_left = nib.load(os.path.join(mask_dir, 'lung_lower_lobe_left.nii.gz')).get_fdata()

            combined_mask = np.zeros_like(mask_upper_right)
            combined_mask[(mask_upper_right > 0) | (mask_middle_right > 0) | (mask_lower_right > 0)] = 1  # Правое легкое
            combined_mask[(mask_upper_left > 0) | (mask_lower_left > 0)] = 2  # Левое легкое

            print("Создаём PNG-проекции...")
            make_projections(input_nifti.get_fdata(), combined_mask, name, PROJECTIONS_DIRECTORY)

            mask_clavicula_left = nib.load(os.path.join(mask_dir, 'clavicula_left.nii.gz')).get_fdata()
            mask_clavicula_right = nib.load(os.path.join(mask_dir, 'clavicula_right.nii.gz')).get_fdata()
            mask_scapula_left = nib.load(os.path.join(mask_dir, 'scapula_left.nii.gz')).get_fdata()
            mask_scapula_right = nib.load(os.path.join(mask_dir, 'scapula_right.nii.gz')).get_fdata()
            mask_rib_left_1 = nib.load(os.path.join(mask_dir, 'rib_left_1.nii.gz')).get_fdata()
            mask_rib_left_6 = nib.load(os.path.join(mask_dir, 'rib_left_6.nii.gz')).get_fdata()
            mask_rib_right_1 = nib.load(os.path.join(mask_dir, 'rib_right_1.nii.gz')).get_fdata()
            mask_rib_right_6 = nib.load(os.path.join(mask_dir, 'rib_right_6.nii.gz')).get_fdata()
            mask_sternum = nib.load(os.path.join(mask_dir, 'sternum.nii.gz')).get_fdata()
            mask_costal_cartilages = nib.load(os.path.join(mask_dir, 'costal_cartilages.nii.gz')).get_fdata()

            combined_mask_bones = np.zeros_like(mask_clavicula_left)#.astype('int')

            combined_mask_bones[mask_scapula_left > 0] = LABEL_MAPPING['scapula_left']
            combined_mask_bones[mask_scapula_right > 0] = LABEL_MAPPING['scapula_right']
            combined_mask_bones[mask_clavicula_left > 0] = LABEL_MAPPING['clavicula_left']
            combined_mask_bones[mask_clavicula_right > 0] = LABEL_MAPPING['clavicula_right']
            combined_mask_bones[mask_rib_left_1 > 0] = LABEL_MAPPING['rib_left_1']
            combined_mask_bones[mask_rib_left_6 > 0] = LABEL_MAPPING['rib_left_6']
            combined_mask_bones[mask_rib_right_1 > 0] = LABEL_MAPPING['rib_right_1']
            combined_mask_bones[mask_rib_right_6 > 0] = LABEL_MAPPING['rib_right_6']
            combined_mask_bones[mask_sternum > 0] = LABEL_MAPPING['sternum']
            combined_mask_bones[mask_costal_cartilages > 0] = LABEL_MAPPING['costal_cartilages']

            make_bones_projections(
                    nib.load(nifti_path).get_fdata(), combined_mask_bones, class_map,
                    system_names=BONE_CLASSES,
                    filename=name,
                    savepath=PROJECTIONS_DIRECTORY,
                    window_center=300, 
                    window_width=1000,
                    smoothing_sigma=0.6,
                    pad_px=8
                )
            
            yield 90, STEP_INFERENCE

            scores = test_AD_each_view.main(RESULTS_DIRECTORY, PROJECTIONS_DIRECTORY)
            anomaly_score = torch.sigmoid(torch.tensor(max(scores)*2.86 - 8.7)).item()

        yield 100, anomaly_score

    finally:
        # Обязательная очистка после завершения
        cleanup_environment()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run DICOM inference for a given study and series.")

    parser.add_argument(
        "--file_path",
        type=str,
        # default="/app/inference/datasets/Датасет/pneumonia_anon.zip",# "/app/inference/datasets/Датасет/pneumotorax_anon.zip",#'/app/inference/datasets/Датасет/norma_anon.zip' 
        # default="/app/inference/datasets/Датасет/pneumotorax_anon.zip",
        default="/app/inference/datasets/Датасет/norma_anon.zip",
        help="Корневая папка с DICOM файлами"
    )
    
    parser.add_argument(
        "--study_id",
        type=str,
        # default="1.2.276.0.7230010.3.1.2.2462171185.19116.1754559747.125",# '1.2.276.0.7230010.3.1.2.2462171185.19116.1754559949.863'  
        # default='1.2.276.0.7230010.3.1.2.2462171185.19116.1754560222.2501',
        default="1.2.276.0.7230010.3.1.2.2462171185.19116.1754559949.863",
        help="StudyInstanceUID для поиска"
    )
    
    parser.add_argument(
        "--series_id",
        type=str,
        # default="1.2.276.0.7230010.3.1.3.2462171185.19116.1754559747.126", # "1.2.276.0.7230010.3.1.3.2462171185.19116.1754559949.864",  # 
        # default="1.2.276.0.7230010.3.1.3.2462171185.19116.1754560222.2502",
        default="1.2.276.0.7230010.3.1.3.2462171185.19116.1754559949.864",
        help="SeriesInstanceUID для поиска"
    )
    
    args = parser.parse_args()
    for x in doInference(args.file_path, args.study_id, args.series_id):
        print(x)
