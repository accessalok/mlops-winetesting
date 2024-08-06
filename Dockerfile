# Dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY app.py app.py
COPY best_model.pkl best_model.pkl

EXPOSE 5000

CMD ["python", "app.py"]
