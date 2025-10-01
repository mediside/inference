import argparse
import os 
import pydicom
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

DEVICE = 'cuda'
MODEL_PATH = 'model.pt'

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
        LoadImaged(keys="image"),
        EnsureChannelFirstd(keys="image"),
        # TODO: с этим на сервере 5070 вылетает в ошибку при transforms
        # Orientationd(keys="image", axcodes="RAS", labels=None),
        Spacingd(keys="image", pixdim=(0.703125, 0.703125, 5.0)),  # как в конфиге
        ScaleIntensityRanged(keys="image", a_min=-1024, a_max=300, b_min=0.0, b_max=1.0, clip=True),
        EnsureTyped(keys="image"),
    ])

    yield 10, STEP_FILE_READ

    with TemporaryFolder(prefix="dicom_") as temp_dir:
        dicom_dir = extract_dicom_series(file_path, study_id, series_id, temp_dir)
        print(dicom_dir)
        yield 20, STEP_LUNG_CHECK

        lungs_flag = solve_lungs(dicom_dir)
        if lungs_flag == 'NO':
            raise ValueError('На КТ снимке не обнаружены легкие')

        yield 30, STEP_PREPROCESSING

        input_image = transforms({'image': dicom_dir})['image'].bfloat16().unsqueeze(0).to(DEVICE)

    yield 40, STEP_INFERENCE_1

    probability_of_pathology = torch.sigmoid(feature_extractor(input_image)['pool'].mean()).item()
    print(probability_of_pathology)
    yield 100, probability_of_pathology

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run DICOM inference for a given study and series.")

    parser.add_argument(
        "--file_path",
        type=str,
        default="/app/inference/datasets/Датасет/norma_anon.zip",  
        help="Корневая папка с DICOM файлами"
    )
    
    parser.add_argument(
        "--study_id",
        type=str,
        default="1.2.276.0.7230010.3.1.2.2462171185.19116.1754559949.863",  
        help="StudyInstanceUID для поиска"
    )
    
    parser.add_argument(
        "--series_id",
        type=str,
        default="1.2.276.0.7230010.3.1.3.2462171185.19116.1754559949.864",  
        help="SeriesInstanceUID для поиска"
    )
    
    args = parser.parse_args()
    for x in doInference(args.file_path, args.study_id, args.series_id):
        print(x)