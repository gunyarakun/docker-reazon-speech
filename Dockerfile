FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    make \
    python3-pip \
    wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN cd / \
  && git clone https://github.com/espnet/espnet \
  && cd espnet/tools \
  && rm -f activate_python.sh && touch activate_python.sh \
  && make TH_VERSION=1.13.1 CUDA_VERSION=12.2

WORKDIR /app
