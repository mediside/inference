import argparse

import os 
import shutil
from pathlib import Path
import pydicom

import numpy as np

import io, zipfile
import torch

import cv2

import matplotlib.pyplot as plt

from totalsegmentator.python_api import totalsegmentator
import nibabel as nib

# from lung_check import solve_lungs
from utils import TemporaryFolder

import dicom2nifti

import importlib
# my_projection = importlib.import_module("VMPR-UAD.Multi_view_projection.my_projection")
# my_inference = importlib.import_module("VMPR-UAD.Segmentation.my_inference")
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
STEP_INFERENCE_1 = 'inference_1'
STEP_INFERENCE_2 = 'inference_2'
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


            # PROJECTION_PATH = "projections_dir_resampled_2"

            nifti_path = os.path.join(NIFTI_FOLDER, f"{name}.nii.gz")
            mask_dir = os.path.join(MASK_DIRECTORY, name)

            dicom2nifti.dicom_series_to_nifti(
                dicom_dir,
                nifti_path,
                reorient_nifti=True
            )

            from nilearn.image import resample_img
            
            input_nifti = nib.load(nifti_path) #.get_fdata()
            target_affine = np.eye(3)  # 1мм изотропный воксель
            input_nifti = resample_img(input_nifti, target_affine=target_affine, interpolation='continuous', copy_header=True)
            
            totalsegmentator(
                input_nifti,  
                mask_dir,
                device='gpu',
                fast=False,
                roi_subset=[
                    'lung_upper_lobe_left', 'lung_lower_lobe_left',
                    'lung_upper_lobe_right', 'lung_middle_lobe_right',
                    'lung_lower_lobe_right'
                ],
                nr_thr_saving=1,
            )

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


            # try: 
            #     lungs_flag = solve_lungs(dicom_dir)
            # except:
            #     print('Что-то пошло не так при проверке')
            #     lungs_flag = 'YES'

            # if lungs_flag == 'NO':
            #     raise ValueError('На КТ снимке не обнаружены легкие')

            yield 30, STEP_PREPROCESSING
            
            # mask_path = my_inference.segment_case_sitk(
            #     dicom_dir,
            #     MASK_DIRECTORY,
            # )
            # my_projection.make_projections(
            #     dicom_dir,
            #     mask_path, 
            #     PROJECTIONS_DIRECTORY
            # )
        
            yield 40, STEP_INFERENCE_1

            scores = test_AD_each_view.main(RESULTS_DIRECTORY, PROJECTIONS_DIRECTORY)
            anomaly_score = torch.sigmoid(torch.tensor(max(scores) - 3)*2).item()

            print("ANOMALY SCORE", anomaly_score)

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
