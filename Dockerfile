FROM python:3.12-slim-trixie

ARG BENCHMARK_COLIBRI_VERSION=2.2.0
ARG COLIBRI_SHA256=b1e2bc8a650b2b216e17357614cfcb4355a83fb9d3077d61f8e1d63beea14d72

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && curl --fail --location --retry 3 \
       "https://github.com/JustVugg/colibri/releases/download/v${BENCHMARK_COLIBRI_VERSION}/colibri-v${BENCHMARK_COLIBRI_VERSION}-linux-x86_64.tar.gz" \
       --output /tmp/colibri.tar.gz \
    && echo "${COLIBRI_SHA256}  /tmp/colibri.tar.gz" | sha256sum --check \
    && mkdir /opt/colibri \
    && tar -xzf /tmp/colibri.tar.gz -C /opt/colibri \
    && rm /tmp/colibri.tar.gz \
    && python /opt/colibri/coli serve --help >/dev/null

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY benchmark.py .
ENV PYTHONUNBUFFERED=1 BENCHMARK_COLIBRI_VERSION=${BENCHMARK_COLIBRI_VERSION}
ENTRYPOINT ["python", "/app/benchmark.py"]
