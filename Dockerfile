FROM python:3.10.6-slim

WORKDIR /root

COPY deploy.py deploy.py

RUN echo "alias ll='ls -al'" >> ~/.bashrc && \
    apt-get update && \
    apt-get install -y vim && \
    pip install kubernetes==26.1.0

# docker run -it --rm \
# -v /etc/kubernetes/admin.conf:/root/.kube/config \
# python:3.10.6-slim-kube /bin/bash
