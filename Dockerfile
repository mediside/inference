FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime
# FROM projectmonai/monai:1.5.1
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .

CMD ["python", "main.py"]
# CMD ["tail", "-f", "/dev/null"] # для отладки
