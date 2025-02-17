# Używamy obrazu bazowego z Pythonem 3.12 i obsługą CUDA
FROM python:3.12-slim

WORKDIR /app

# Pobranie i instalacja pget
RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.8.2/pget_linux_x86_64" && chmod +x /usr/local/bin/pget

# Kopiujemy pliki projektu
COPY . .

# Instalujemy pip i zależności
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "predict.py"]
