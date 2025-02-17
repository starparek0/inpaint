FROM python:3.12-slim

WORKDIR /app

# Pobieramy pget
RUN curl -o /usr/local/bin/pget -L "https://github.com/replicate/pget/releases/download/v0.8.2/pget_linux_x86_64" && chmod +x /usr/local/bin/pget

# Pobieramy model już na etapie budowania obrazu
RUN curl -o /src/model.tar "https://weights.replicate.delivery/default/black-forest-labs/FLUX.1-dev/files.tar" \
    && mkdir -p /src/FLUX.1-dev \
    && tar -xf /src/model.tar -C /src/FLUX.1-dev \
    && rm /src/model.tar

COPY . .

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["python", "predict.py"]
