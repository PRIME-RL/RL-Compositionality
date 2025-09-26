FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu20.04

COPY . /app
WORKDIR /app

RUN pip3 install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124

