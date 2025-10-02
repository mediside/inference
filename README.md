# inference-vmpr — подробное README

> Документ с описанием проекта, структурой папок, инструкцией по запуску и подробным описанием подхода инференса.

---

## Краткое описание

`inference-vmpr` — это Python-процесс для выполнения инференса на данных (в основном DICOM / NIfTI томограммы). Проект запускает gRPC-сервис, который получает запросы на инференс и возвращает поток прогресса и итоговый результат (вероятность патологии). Внутри реализован пайплайн чтения изучения, проверки наличия легких, сегментации легких (lungmask / ResUNet), мультивидовая проекция и шаги для детекции/анализ аномалий (модуль VMPR-UAD).

---

## Быстрый старт

### Установка зависимостей

```bash
pip install -r requirements.txt
```

> Требуемые библиотеки перечислены в `requirements.txt` (grpcio, protobuf, torch, monai, pydicom, lungmask и т.д.).

### Запуск сервиса

```bash
python main.py
```

По умолчанию gRPC слушает на `localhost:30042` (см. `main.py`).

### Запуск в Docker

Сборка и запуск контейнера (в Dockerfile уже указан базовый образ и команда запуска):

```bash
docker build -t inference-vmpr .
docker run --rm -p 30042:30042 inference-vmpr
```

---

## Содержание репозитория (структура папок)

Ниже — дерево проекта с пояснениями, что где находится.

```
inference-vmpr/
├─ .gitignore
├─ Dockerfile                 # образ, команда запуска
├─ Makefile                   # helper: compile-proto
├─ README.md                  # исходный краткий readme
├─ main.py                    # gRPC сервер, точка входа
├─ inference.py               # главный пайплайн: doInference()
├─ lung_check.py              # проверка на наличие легких (lungmask)
├─ utils.py                   # утилиты (временные папки, загрузка DICOM/NIfTI -> numpy)
├─ requirements.txt
├─ proto/
│  ├─ inference.proto         # описание gRPC API
│  ├─ inference_pb2.py        # сгенерированный protobuf (py)
│  └─ inference_pb2_grpc.py   # сгенерированный gRPC код
└─ VMPR-UAD/                   # модули VMPR-UAD: проекция, сегментация, AD
   ├─ Multi_view_projection/
   │  └─ my_projection.py     # функции мультивидовой проекции
   ├─ Segmentation/
   │  ├─ my_inference.py      # обёртки для сегментации случая
   │  └─ lungmask/
   │     ├─ mask.py           # персональная логика работы с lungmask
   │     ├─ resunet.py        # модель ResUNet (архитектура и inference helpers)
   │     └─ utils.py          # вспом. функции для сегментации
   └─ Anomaly_detection/
      ├─ generate_3d_map.py   # сборка 3D map для аномалий
      └─ sampling_methods/    # вспом. для выборки сэмплов
         ├─ kcenter_greedy.py
         └─ sampling_def.py
```

> Файлы `proto/*.py` сгенерированы из `inference.proto`. Если нужно пересобрать протобуфы, используйте Makefile:
>
> ```bash
> make compile-proto
> # или
> python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. proto/*.proto
> ```

---

## Что делает каждая часть — подробное описание

### `main.py` — gRPC сервер

`main.py` реализует gRPC-сервис `Inference` с методом `DoInference`. Когда приходит `InferenceRequest` (поля `file_path`, `study_id`, `series_id`), сервер вызывает `inference.doInference(...)`, который реализован как генератор. Генератор по мере выполнения возвращает сообщения прогресса (процент + ключевой шаг) и в конце — `Result` с `probability_of_pathology`.

**Протокол:**

* `InferenceRequest` — вход: `file_path`, `study_id`, `series_id`.
* `InferenceResponse` — stream с `oneof { Progress | Result }`.
* `Progress` содержит `percent` (int32) и `step` (string).
* `Result` содержит `probability_of_pathology` (float).

Это позволяет клиенту подписаться на поток и обновлять прогресс по мере вычислений.

### `inference.py` — основной пайплайн

В файле реализована функция `doInference(file_path, study_id, series_id)`, которая:

1. Отправляет начальные шаги прогресса (STEP_START, STEP_FILE_READ и т.д.).
2. Распаковывает или извлекает DICOM/NIfTI данные в временную папку (`TemporaryFolder` из `utils.py`).
3. Вызывает проверку `solve_lungs` из `lung_check.py` — проверяет, есть ли в данных легкие (сегментация/метаданные).
4. Если легкие присутствуют, запускает этапы предобработки и сегментации (использует VMPR-UAD/Segmentation и `lungmask`).
5. Применяет мультивидовую проекцию (`VMPR-UAD/Multi_view_projection/my_projection.py`) и генерацию 3D карт (`VMPR-UAD/Anomaly_detection/generate_3d_map.py`).
6. Возвращает итоговую оценку (в текущем шаблоне — одно число `probability_of_pathology`).

Ключевые функции в `inference.py`:

* `extract_dicom_series(file_path, study_id, series_id, temp_dir)` — ищет и извлекает нужную серию DICOM из zip или каталога и возвращает путь к распакованной серии.
* `doInference(...)` — генератор прогресса и результата.

**Шаги (STEP_...):** `start`, `file_read`, `lung_check`, `preprocessing`, `finish`.

### `utils.py`

Содержит вспомогательные классы/функции:

* `TemporaryFolder` — контекстный менеджер для временной директории с автоматическим удалением.
* `load_arr(input_path)` — функция загрузки изображения из DICOM (через SimpleITK + pydicom) или NIfTI (nibabel) и приведение к единому ndarray формата `(X,Y,Z)`.

`load_arr` приводит ориентацию к RAS (через SimpleITK) и нормализует/приводит к float32.

### `lung_check.py`

Использует `lungmask` (LMInferer) и некоторые эвристики по `meta` из MONAI/LoadImage для того, чтобы определить:

* есть ли в изучении легкие (возвращает `"YES"`/`"NO"`).

Используется в пайплайне, чтобы отсекать исследования, не содержащие легких.

### VMPR-UAD / Multiview / Segmentation / Anomaly_detection

Эти модули реализуют логику работы с мультивидовой проекцией и детекцией аномалий:

* `Segmentation/my_inference.py` и `Segmentation/lungmask/*` — обёртки вокруг lungmask + собственных моделей (ResUNet) для генерации масок легких и подготовки данных.
* `Multi_view_projection/my_projection.py` — функции для обрезки области легких, проекций в разные плоскости и объединения видов.
* `Anomaly_detection/generate_3d_map.py` — по картам и проекциям создаёт 3D-представление поражений; используется далее для выборки и принятия решения.
* `Anomaly_detection/sampling_methods` — вспомогательные алгоритмы (например k-center greedy) для отбора репрезентативных срезов/фреймов.

## Пример gRPC-клиента (Python)

Ниже — минимальный пример клиента, который отправляет запрос и печатает прогресс и результат:

```python
import grpc
from proto import inference_pb2, inference_pb2_grpc

channel = grpc.insecure_channel('localhost:30042')
stub = inference_pb2_grpc.InferenceStub(channel)

req = inference_pb2.InferenceRequest(
    file_path='/path/to/dataset.zip',
    study_id='1.2....studyid...',
    series_id='1.2....seriesid...'
)

for resp in stub.DoInference(req):
    # InferenceResponse: oneof { progress | result }
    if resp.HasField('progress'):
        print(f"Progress: {resp.progress.percent}% - {resp.progress.step}")
    elif resp.HasField('result'):
        print(f"Result probability: {resp.result.probability_of_pathology}")
```

> Примечание: если `proto/*.py` не сгенерированы, сначала выполните `make compile-proto`.


Рекоммендация:

* Создайте `models/` в корне, поместите в неё необходимые `.pth` или `.pt` файлы.
* В `Segmentation/resunet.py` и в местах, где используются веса, добавьте аргумент/конфиг с путём к весам.

## Типовой поток данных (подробно)

1. **Получение запроса** — gRPC получает `file_path, study_id, series_id`.
2. **Распаковка / чтение** — `extract_dicom_series` распаковывает zip (если указан zip) или читает директорию, находит все DICOM-файлы с нужным `StudyInstanceUID` и `SeriesInstanceUID`, и сохраняет их во временную папку.
3. **Проверка легких** — `solve_lungs` использует `lungmask` (LMInferer) и метаинформацию: если сегмент/мета указывает на легкие, продолжаем, иначе возвращаем `probability=0` или специальный код.
4. **Загрузка в unified numpy** — `load_arr` приводит данные в формат `(X,Y,Z)` float32 и стандартную ориентацию (RAS).
5. **Сегментация** — если нужно, код вызывает `segment_case_sitk` (в `my_inference.py`), генерирует маски `.nii.gz` и/или возвращает маску как numpy.
6. **Проекция / агрегирование** — `my_projection.py` обрезает маску до полезной области, создаёт мультивидовые проекции (несколько входных изображений/видов) и нормализует их.
7. **Аномалия / scoring** — `generate_3d_map.py` и связанные модули собирают признаки/карты и возвращают скалярную вероятность патологии (в данном шаблоне). В реальной системе этот шаг заменяется обученной моделью детекции/классификации.
8. **Возврат результата** — `doInference` посылает `Result` с полем `probability_of_pathology`.

## Контакты / кто что делает

* Александр Югай (ML) - @AleksandrY99
* Максим Лопатин (FullStack) - @maksimioX
* Ирэна Гуреева (ML) - @thdktgdk
* Окунев Даниил (ML) - @danzzzll
* Скрипкин Матвей (ML) - @barracuda049

---
