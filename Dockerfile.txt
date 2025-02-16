FROM python:3.8-slim

# Instalacja zależności systemowych
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Kopiujemy plik z zależnościami Pythona i instalujemy je
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip && pip install -r /tmp/requirements.txt

# Kopiujemy wszystkie pliki projektu do obrazu
COPY . /app
WORKDIR /app

# Polecenie startowe - Cog wykryje klasę Predictor z pliku cog.yaml
CMD ["python", "-m", "cog"]
