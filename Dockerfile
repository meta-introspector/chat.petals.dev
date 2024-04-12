FROM python:3.10-slim-bullseye

RUN apt update && apt install -y libopenblas-dev ninja-build build-essential wget git
RUN python -m pip install --upgrade pip pytest cmake scikit-build setuptools

WORKDIR /usr/src/app/

COPY requirements.txt ./


RUN pip install --no-cache-dir -r ./requirements.txt --upgrade pip

ADD . /usr/src/app/

CMD gunicorn app:app --bind 0.0.0.0:5000 --worker-class gthread --threads 100 --timeout 1000