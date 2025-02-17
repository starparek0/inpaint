# Używamy obrazu bazowego z Pythonem 3.12 i obsługą CUDA
FROM python:3.12-slim

WORKDIR /app

# Pobranie wag w fazie budowania, aby uniknąć długiego setupu
RUN curl -o /src/model.tar "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar" \
    && mkdir -p /src/FLUX.1-dev \
    && tar -xf /src/model.tar -C /src/FLUX.1-dev \
    && rm /src/model.tar

# Kopiujemy pliki projektu
COPY . .

# Instalujemy pip i zależności
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "predict.py"]
