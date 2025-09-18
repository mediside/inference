FROM python:3.11.13-alpine3.22
WORKDIR /app
COPY . .
CMD ["python", "main.py"]