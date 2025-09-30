import argparse

import shutil
from pathlib import Path
import tempfile
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


DEVICE = 'cuda'
MODEL_PATH = 'model.pt'

STEP_START = 'start' # скрипт жив и начал инференс
STEP_FILE_READ = 'file_read' # скрипт прочитал файл
STEP_LUNG_CHECK = 'lung_check'
STEP_PREPROCESSING = 'preprocessing' # скрипт закончил препроцессинг
STEP_INFERENCE_1 = 'inference_1'
STEP_INFERENCE_2 = 'inference_2'
STEP_FINISH = 'finish' # скрипт закончил инференс


def load_from_zip(zip_path, study_id, series_id):
    dicoms = []
    with zipfile.ZipFile(zip_path, "r") as zf:
        for name in zf.namelist():
            try:
                ds = pydicom.dcmread(io.BytesIO(zf.read(name)), force=True)
                if (getattr(ds, "StudyInstanceUID", None) == study_id and
                    getattr(ds, "SeriesInstanceUID", None) == series_id):
                    dicoms.append(ds)
            except Exception:
                continue
    return dicoms  # список pydicom.Dataset

def dicoms_to_array(dicom_list):
    """
    Принимает список pydicom.Dataset и возвращает np.array тома + мета словарь.
    """
    # Сортируем срезы по InstanceNumber
    dicom_list = sorted(dicom_list, key=lambda x: int(getattr(x, "InstanceNumber", 0)))
    
    # Строим numpy массив
    volume = np.stack([d.pixel_array for d in dicom_list], axis=0)
    
    # Мета информация
    meta = {
        "spacing": (
            float(getattr(dicom_list[0], "SliceThickness", 1.0)),
            float(getattr(dicom_list[0], "PixelSpacing", [1.0, 1.0])[0]),
            float(getattr(dicom_list[0], "PixelSpacing", [1.0, 1.0])[1]),
        ),
        "affine": None,  # MONAI сможет сам построить affine из spacing и orientation
        "original_shape": volume.shape,
        "dtype": volume.dtype,
    }
    
    return volume, meta

def doInference(file_path: str, study_id: str, series_id: str):
    print(f'filepath: {file_path}, study_id: {study_id}, series_id: {series_id}')
    yield 0, STEP_START
    yield 10, STEP_FILE_READ

    dicoms = load_from_zip(file_path, study_id, series_id)
    volume, meta = dicoms_to_array(dicoms)

    data_dict = {
        "image": MetaTensor(torch.tensor(volume[None, ...], dtype=torch.float32), meta)
    }

    yield 20, STEP_LUNG_CHECK

    lungs_flag = solve_lungs(volume, meta)
    if lungs_flag == 'NO':
        raise ValueError('На КТ снимке не обнаружены легкие')

    #  === Backbone (3D ResNet50) ===
    backbone = resnet50(
        spatial_dims=3,
        n_input_channels=1,
        conv1_t_stride=[2, 2, 1],
        conv1_t_size=[7, 7, 7],
    )

    # === Feature extractor (ResNet50 + FPN) ===
    feature_extractor = resnet_fpn_feature_extractor(
        backbone, 3, False, [1, 2], None
    )

    # === RetinaNet ===
    model = RetinaNet(
        spatial_dims=3,
        num_classes=1,
        num_anchors=3,
        feature_extractor=feature_extractor,
        size_divisible=[16, 16, 8],
        use_list_output=False,
    )

    # === Подгружаем веса ===
    ckpt = torch.load(MODEL_PATH)
    model.load_state_dict(ckpt)
    feature_extractor = model.feature_extractor.bfloat16().to(DEVICE).eval()

    transforms = Compose([
        # LoadImaged(keys="series_path"),
        EnsureChannelFirstd(keys="image"),
        Orientationd(keys="image", axcodes="RAS", labels=None),
        Spacingd(keys="image", pixdim=(0.703125, 0.703125, 5.0)),  # как в конфиге
        ScaleIntensityRanged(keys="image", a_min=-1024, a_max=300, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys="image"),
    ])

    yield 30, STEP_PREPROCESSING

    input_image = transforms(data_dict)['image'].bfloat16().unsqueeze(0).to(DEVICE)

    yield 40, STEP_INFERENCE_1

    probability_of_pathology = torch.sigmoid(feature_extractor(input_image)['pool'].mean())
    print(probability_of_pathology)
    yield 100, probability_of_pathology

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run DICOM inference for a given study and series.")

    parser.add_argument(
        "--file_path",
        type=str,
        default="/app/inference/datasets/MosMedData-LDCT-LUNGCR-type I-v 1/studies/1.2.643.5.1.13.13.12.2.77.8252.00001007020103130905041401130706",  
        help="Корневая папка с DICOM файлами"
    )
    
    parser.add_argument(
        "--study_id",
        type=str,
        default="1.2.643.5.1.13.13.12.2.77.8252.00001007020103130905041401130706",  
        help="StudyInstanceUID для поиска"
    )
    
    parser.add_argument(
        "--series_id",
        type=str,
        default="1.2.643.5.1.13.13.12.2.77.8252.02110901050806091404100511030202",  
        help="SeriesInstanceUID для поиска"
    )
    
    args = parser.parse_args()
    for x in doInference(args.file_path, args.study_id, args.series_id):
        print(x)