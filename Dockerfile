FROM nvidia/cuda:13.0.2-cudnn-runtime-ubuntu24.04

WORKDIR /app

COPY environment.yml /app/environment.yml

RUN apt-get update && apt-get install -y \
    wget \
    git \
    bzip2 \
    ca-certificates \
 && rm -rf /var/lib/apt/lists/*

ENV CONDA_DIR=/opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh && \
    bash /tmp/miniconda.sh -b -p $CONDA_DIR && \
    rm /tmp/miniconda.sh && \
    conda clean -afy

SHELL ["bash", "-lc"]

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

RUN conda env create -f /app/environment.yml && \
    conda clean -afy

RUN /opt/conda/envs/deep/bin/pip install --no-cache-dir \
        torch torchvision \
        --index-url https://download.pytorch.org/whl/cu130


# RUN conda activate deep && \
#     pip install --no-cache-dir \
#         torch torchvision \
#         --index-url https://download.pytorch.org/whl/cu130

ENV CONDA_DEFAULT_ENV=deep
ENV PATH=/opt/conda/envs/${CONDA_DEFAULT_ENV}/bin:$PATH

RUN git clone -b Docker https://github.com/karol-nowinski/CBCT_bone_segmentation.git /app/cbct_bone_segmentation

CMD ["bash"]