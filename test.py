import zipfile
import io
import pydicom

with open("../apiservice/./researches/f0591175-56e2/pneumonia_anon.zip", "rb") as f:
    zip_data = f.read()

zip_buffer = io.BytesIO(zip_data)

with zipfile.ZipFile(zip_buffer, "r") as archive:
    for file_name in archive.namelist():
        with archive.open(file_name) as file:
            ds = pydicom.dcmread(file, stop_before_pixels=True, force=True)
            print(f"Файл: {file_name}, StudyInstanceUID: {getattr(ds, "StudyInstanceUID", None)}")