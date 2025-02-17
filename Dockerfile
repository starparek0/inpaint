# Używamy obrazu z Python 3.12, zgodnie z deklaracją w cog.yaml
FROM python:3.12-slim

WORKDIR /app

# Kopiujemy plik z zależnościami
COPY requirements.txt .

RUN pip install --upgrade pip && pip install -r requirements.txt

# Kopiujemy resztę plików projektu
COPY . .

CMD ["python", "predict.py"]
