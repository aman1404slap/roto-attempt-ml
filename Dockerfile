# One image, three jobs. Which job a container does is decided by the command ECS overrides
# with -- build a dataset, train a rung, score a run -- so there is one thing to build, push
# and keep in step with the code.
#
# Base pinned to the same CUDA 13.0 line the sibling services run and the same torch the
# published numbers were measured on (migration note 5.6). Changing either means re-measuring
# the seed spread before the next result is quoted.
FROM pytorch/pytorch:2.12.0-cuda13.0-cudnn9-runtime AS base

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

ENV PYTORCH_ALLOC_CONF=expandable_segments:True
ENV OMP_NUM_THREADS=1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,video,utility

# roto.exr sets this itself, but OpenCV reads it at import time in some builds and the
# archive's EXRs are undecodable without it. Belt and braces: a shot that will not decode
# fails a long way into a build.
ENV OPENCV_IO_ENABLE_OPENEXR=1

# The training code is a plain package under src/ and is imported, not installed, exactly as
# it is in a shell. One less thing that can differ between the image and a developer's box.
ENV PYTHONPATH=/code/src

ARG TORCH_WHL_INDEX_URL=https://download.pytorch.org/whl/cu130

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg g++ htop wget gnupg ca-certificates libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code/

# Two pip stages so a service dependency bump does not rebuild the training stack.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY requirements_inference.txt .
RUN pip install --no-cache-dir -r requirements_inference.txt
RUN pip install --no-cache-dir --index-url ${TORCH_WHL_INDEX_URL} torch==2.12.0

COPY . .

# No CMD. Every container is launched with an explicit command override, so there is no
# default job and nothing starts training by accident:
#
#   python3 ./manage.py train_run --run-id <uuid>
#   python3 ./manage.py build_dataset --dataset-version v004 --tier tier1
#   python3 ./manage.py score_run --rung s3a --seeds 1 2
#
# The web service runs gunicorn instead:
#
#   gunicorn roto_app.wsgi:application --bind 0.0.0.0:5000 --timeout 120
