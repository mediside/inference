import argparse

import os 
import shutil
from pathlib import Path
import pydicom

import numpy as np

import io, zipfile
import torch

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, EnsureTyped
)

from monai.networks.nets import resnet50
from monai.apps.detection.networks.retinanet_network import RetinaNet
from monai.apps.detection.networks.retinanet_network import resnet_fpn_feature_extractor
from monai.data import PydicomReader, MetaTensor

from lung_check import solve_lungs
from utils import TemporaryFolder

import importlib
my_projection = importlib.import_module("VMPR-UAD.Multi_view_projection.my_projection")
my_inference = importlib.import_module("VMPR-UAD.Segmentation.my_inference")

DEVICE = 'cuda'
MODEL_PATH = 'model.pt'

MASK_DIRECTORY = 'mask_directory'
PROJECTIONS_DIRECTORY = 'projections_directory'

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


def doInference(file_path: str, study_id: str, series_id: str):
    print(f'filepath: {file_path}, study_id: {study_id}, series_id: {series_id}')
    yield 0, STEP_START

    yield 10, STEP_FILE_READ

    with TemporaryFolder(prefix="dicom_") as temp_dir:
        dicom_dir = extract_dicom_series(file_path, study_id, series_id, temp_dir)

        yield 20, STEP_LUNG_CHECK

        # try: 
        #     lungs_flag = solve_lungs(dicom_dir)
        # except:
        #     print('Что-то пошло не так при проверке')
        #     lungs_flag = 'YES'

        # if lungs_flag == 'NO':
        #     raise ValueError('На КТ снимке не обнаружены легкие')

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

        # input_image = transforms({'image': dicom_dir})['image'].bfloat16().unsqueeze(0).to(DEVICE)

    yield 40, STEP_INFERENCE_1

    probability_of_pathology = torch.rand(1).item() #torch.sigmoid(feature_extractor(input_image)['pool'].mean()).item()
    print(probability_of_pathology)
    yield 100, probability_of_pathology

    
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