import shutil
import tempfile
from pathlib import Path


class TemporaryFolder:
    def __init__(self, prefix="temp_folder_"):
        self.prefix = prefix
        self.path = None

    def __enter__(self):
        # создаём временную папку
        self.path = Path(tempfile.mkdtemp(prefix=self.prefix))
        return self.path

    def __exit__(self, exc_type, exc_value, traceback):
        # удаляем папку со всем содержимым
        if self.path and self.path.exists():
            shutil.rmtree(self.path)
