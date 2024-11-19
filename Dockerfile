FROM python:3.8 as builder

RUN mkdir /install
WORKDIR /install

# Copy ca.cer (certificate authority) if it exists. Necessary in a SSL decrypt evironment.
COPY requirements.txt ca.cer* /

RUN apt-get update -y && \
    apt-get install -y git && \
    (test ! -f /ca.cer || git config --global http.sslCAInfo /ca.cer) && \
    (test ! -f /ca.cer || pip config set global.cert /ca.cer) && \
    pip install --prefix=/install -r /requirements.txt

FROM python:3.8-slim

ARG version_number
ARG commit_sha

ENV VERSION_NUMBER=$version_number
ENV COMMIT_SHA=$commit_sha
LABEL COMMIT_SHA=$commit_sha
LABEL VERSION_NUMBER=$version_number

COPY --from=builder /install /usr/local
COPY bottomdetection /app/bottomdetection

ENV PYTHONPATH "${PYTHONPATH}:/app"

WORKDIR /app

CMD ["sh", "-c", "python3 -u /app/bottomdetection/docker_main.py >> /dataout/log_bottom.txt 2>&1"]
