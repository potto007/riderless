# riderless, CUDA 13 flavor: the package plus an already-built worker bundle.
#
# The image compiles nothing. It expects the cuda13 tarball the release
# workflow produced to be present in the build context, and unpacks it where
# ApiConfig already looks. Model weights are never baked in: mount the
# directory holding your GGUF at /models.
#
#   docker build -f docker/Dockerfile \
#     --build-arg BUNDLE=dist/riderless-worker-0.2.0-linux-x64-cuda13.tar.gz .
#   docker run --gpus all -p 8090:8090 -v /path/to/models:/models:ro <image>
#
# The runtime base carries the CUDA runtime, and the bundle carries its own
# libcudart/libcublas/libcublasLt, so the only host requirement is an NVIDIA
# driver new enough for CUDA 13 (580.65 or newer) plus the container toolkit.
FROM nvidia/cuda:13.4.1-runtime-ubuntu24.04

ARG BUNDLE

# python3 is the interpreter; the venv keeps pip out of the system install,
# which Ubuntu 24.04 marks externally managed. libgomp1 is the one system
# library the bundle does not carry: ggml's CPU backend links OpenMP, and the
# CUDA runtime image ships without it.
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/riderless
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
RUN python3 -m venv "${VIRTUAL_ENV}"

WORKDIR /srv/riderless
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY riderless ./riderless
RUN pip install --no-cache-dir .

# `build/api-worker/build/riderless-worker` and `build/api-worker/build.json`
# are ApiConfig's defaults, so serve needs no worker flags.
# ADD extracts a local tar archive into the destination, so the bundle lands
# unpacked and the tarball itself never becomes a layer.
ADD ${BUNDLE} build/api-worker/

# One GGUF, mounted read-only. The name is ApiConfig's default; override it
# with --model-path on the command line if yours is named differently.
VOLUME ["/models"]
ENV RIDERLESS_MODEL=/models/gemma-4-26B-A4B-it-UD-Q4_K_XL.gguf

EXPOSE 8090
# One request at a time and no auth: put this behind something (see the
# README's Limitations) before exposing it beyond a trusted network.
ENTRYPOINT ["/bin/sh", "-c", \
  "exec python -m riderless.api.cli serve --gpu --host 0.0.0.0 --port 8090 \
   --model-path \"$RIDERLESS_MODEL\" \"$@\"", "riderless"]
