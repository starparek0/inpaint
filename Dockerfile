FROM python:3.12

WORKDIR /app

COPY . .

RUN pip install --upgrade pip
RUN pip install /tmp/cog-0.13.7-py3-none-any.whl 'pydantic<2'

# Inne polecenia