import argparse

import os 
import shutil
from pathlib import Path
import pydicom

import numpy as np

import io, zipfile
import torch

from lung_check import solve_lungs
from utils import TemporaryFolder

import importlib
my_projection = importlib.import_module("VMPR-UAD.Multi_view_projection.my_projection")
my_inference = importlib.import_module("VMPR-UAD.Segmentation.my_inference")
test_AD_each_view = importlib.import_module("VMPR-UAD.Anomaly_detection.test_AD_each_view")


DEVICE = 'cuda'
MODEL_PATH = 'model.pt'

MASK_DIRECTORY = 'mask_directory'
PROJECTIONS_DIRECTORY = 'projections_directory'
RESULTS_DIRECTORY = 'results'

STEP_START = 'start' # скрипт жив и начал инференс
STEP_FILE_READ = 'file_read' # скрипт прочитал файл
STEP_LUNG_CHECK = 'lung_check'
STEP_PREPROCESSING = 'preprocessing' # скрипт закончил препроцессинг
STEP_INFERENCE_1 = 'inference_1'
STEP_INFERENCE_2 = 'inference_2'
STEP_FINISH = 'finish' # скрипт закончил инференс


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
    directories = [MASK_DIRECTORY, PROJECTIONS_DIRECTORY, RESULTS_DIRECTORY]
    
    for dir_path in directories:
        # Создаем директорию если не существует
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        
        # Очищаем директорию
        clean_directory(dir_path)
        print(f"Подготовлена директория: {dir_path}")

def cleanup_environment():
    """Очищает все рабочие директории"""
    directories = [MASK_DIRECTORY, PROJECTIONS_DIRECTORY, RESULTS_DIRECTORY]
    
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

            try: 
                lungs_flag = solve_lungs(dicom_dir)
            except:
                print('Что-то пошло не так при проверке')
                lungs_flag = 'YES'

            if lungs_flag == 'NO':
                raise ValueError('На КТ снимке не обнаружены легкие')

            yield 30, STEP_PREPROCESSING
            
            mask_path = my_inference.segment_case_sitk(
                dicom_dir,
                MASK_DIRECTORY,
            )
            my_projection.make_projections(
                dicom_dir,
                mask_path, 
                PROJECTIONS_DIRECTORY
            )
        
            yield 40, STEP_INFERENCE_1

            scores = test_AD_each_view.main(RESULTS_DIRECTORY, PROJECTIONS_DIRECTORY)
            anomaly_score = max(scores)

        yield 100, anomaly_score

    finally:
        # Обязательная очистка после завершения
        cleanup_environment()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run DICOM inference for a given study and series.")

    parser.add_argument(
        "--file_path",
        type=str,
        default="/app/inference/datasets/Датасет/pneumotorax_anon.zip",  
        help="Корневая папка с DICOM файлами"
    )
    
    parser.add_argument(
        "--study_id",
        type=str,
        default="1.2.276.0.7230010.3.1.2.2462171185.19116.1754560222.2501",  
        help="StudyInstanceUID для поиска"
    )
    
    parser.add_argument(
        "--series_id",
        type=str,
        default="1.2.276.0.7230010.3.1.3.2462171185.19116.1754560222.2502",  
        help="SeriesInstanceUID для поиска"
    )
    
    args = parser.parse_args()
    for x in doInference(args.file_path, args.study_id, args.series_id):
        print(x)