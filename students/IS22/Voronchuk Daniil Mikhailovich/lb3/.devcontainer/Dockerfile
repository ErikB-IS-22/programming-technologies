FROM mcr.microsoft.com/devcontainers/python:1-3.12-bullseye

ENV PYTHONUNBUFFERED=1
#     CUDA_HOME=/usr/local/cuda \
#     PATH=/usr/local/cuda/bin:${PATH} \
#     LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}

# CUDA репо + toolkit
# RUN apt-get update && apt-get install -y --no-install-recommends \
#       wget gnupg ca-certificates && \
#     wget https://developer.download.nvidia.com/compute/cuda/repos/debian11/x86_64/cuda-keyring_1.1-1_all.deb && \
#     dpkg -i cuda-keyring_1.1-1_all.deb && rm cuda-keyring_1.1-1_all.deb && \
#     apt-get update && \
#     apt-get install -y --no-install-recommends cuda-runtime-12-4 cuda-libraries-12-4 && \
#     rm -rf /var/lib/apt/lists/*

WORKDIR /workspaces

# зависимости для лабы
COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r /tmp/requirements.txt && rm /tmp/requirements.txt

CMD ["bash"]
