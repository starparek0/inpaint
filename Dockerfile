FROM python:3.11-slim

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install /tmp/cog-0.13.7-py3-none-any.whl 'pydantic==1.10.7'

# Inne polecenia