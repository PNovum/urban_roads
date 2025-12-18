FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY refresh.py .
COPY train_model.py .
COPY run.py .

CMD ["python", "run.py"]
