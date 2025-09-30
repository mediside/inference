import argparse

import shutil
from pathlib import Path
import tempfile
import pydicom
import zipfile
import io

import torch

from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, EnsureTyped
)

from monai.networks.nets import resnet50
from monai.apps.detection.networks.retinanet_network import RetinaNet
from monai.apps.detection.networks.retinanet_network import resnet_fpn_feature_extractor

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


def get_dir(file_path: str, study_id: str, series_id: str) -> str:
    """
    Находит все DICOM файлы с указанными study_id и series_id внутри file_path,
    копирует их во временную папку и возвращает путь к ней.
    """
    temp_dir = './tmp/'
    
    file_path = Path(file_path)
    if not file_path.exists():
        raise ValueError(f"Путь {file_path} не существует")
    else:
        print(f"Путь {file_path} существует")
    
    selected_files = []

    # Открываем ZIP-файл с диска в бинарном режиме и читаем содержимое в байты
    with open(file_path, "rb") as f:
        zip_data = f.read()
    
    zip_buffer = io.BytesIO(zip_data)
    with zipfile.ZipFile(zip_buffer, "r") as archive:
        for file_name in archive.namelist():
            with archive.open(file_name) as file:
                try:
                    ds = pydicom.dcmread(file, stop_before_pixels=True, force=True)
                    if (getattr(ds, "StudyInstanceUID", None) in study_id and
                        getattr(ds, "SeriesInstanceUID", None) in series_id):
                        selected_files.append(file)
                        print("PATHPATH", './tmp/' + file.name.split('/')[-1])
                        with open(temp_dir + file.name.split('/')[-1], "wb") as out_file:
                            out_file.write(file.read())
                except Exception as e:
                    print(e)
                    continue # Пропускаем файлы, которые не являются корректными DICOM
    
    if not selected_files:
        raise ValueError(f"Не найдено файлов для study_id={study_id}, series_id={series_id}")
    
    # Копируем во временную папку
    
    print(f"Скопировано {len(selected_files)} файлов в {temp_dir}")
    return temp_dir


def doInference(file_path: str, study_id: str, series_id: str):
    file_path = '../apiservice/' + file_path

    print(f'filepath: {file_path}, study_id: {study_id}, series_id: {series_id}')
    yield 0, STEP_START
    yield 10, STEP_FILE_READ

    dycom_dir = get_dir(file_path, study_id, series_id)

    print("DICOMDIR", dycom_dir)

    yield 20, STEP_LUNG_CHECK

    lungs_flag = solve_lungs(dycom_dir)
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
        LoadImaged(keys="series_path"),
        EnsureChannelFirstd(keys="series_path"),
        Orientationd(keys="series_path", axcodes="RAS", labels=None),
        Spacingd(keys="series_path", pixdim=(0.703125, 0.703125, 5.0)),  # как в конфиге
        ScaleIntensityRanged(keys="series_path", a_min=-1024, a_max=300, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys="series_path"),
    ])

    yield 30, STEP_PREPROCESSING

    input_image = transforms({'series_path': dycom_dir})['series_path'].bfloat16().unsqueeze(0).to(DEVICE)
    shutil.rmtree(dycom_dir)  # удаляем папку полностью

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