import time
import random

STEP_START = 'start' # скрипт жив и начал инференс
STEP_FILE_READ = 'file_read' # скрипт прочитал файл
STEP_PREPROCESSING = 'preprocessing' # скрипт закончил препроцессинг
STEP_INFERENCE_1 = 'inference_1'
STEP_INFERENCE_2 = 'inference_2'
STEP_FINISH = 'finish' # скрипт закончил инференс

def doInference(file_path: str):
    print('Путь к файлу (ZIP-архив):', file_path)
    yield 0, STEP_START
    
    time.sleep(0.5) # читаем файл
    yield 10, STEP_FILE_READ
    
    time.sleep(1) # препроцессинг
    yield 25, STEP_PREPROCESSING

    time.sleep(2) # инференс, этап 1
    yield 42, STEP_INFERENCE_1

    time.sleep(0.5) # инференс, этап 2
    yield 77, STEP_INFERENCE_2

    time.sleep(1) # инференс закончен
    probability_of_pathology = random.random()
    yield 100, probability_of_pathology

