# Pinned to torch 2.5.1+cu121 to match the validated env (H-Optimus + timm 0.9.16).
FROM --platform=linux/amd64 docker.io/pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime AS reg2026_algorithm_amd64
# PyTorch + CUDA base so GPU is available at inference time.

ENV PYTHONUNBUFFERED=1
# Ensure imports like "from core import ..." and "import trident" work from /opt/app
ENV PYTHONPATH=/opt/app
# No network at run time -> force offline foundation-model loading
ENV HF_HUB_OFFLINE=1
ENV TRANSFORMERS_OFFLINE=1

# Native libraries for slide IO (as root, before dropping privileges):
#   libopenslide0 -> openslide-python ; libvips42 -> pyvips (AdobeDeflate conversion)
RUN apt-get update \
    && apt-get install -y --no-install-recommends libopenslide0 libvips42 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app
ENV PATH=/home/user/.local/bin:${PATH}

COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Platform entrypoint + helpers
COPY --chown=user:user core.py      /opt/app/
COPY --chown=user:user inference.py /opt/app/
# Our code: interface handlers + attribute model + composer + glue + single-slide pipeline
COPY --chown=user:user src/         /opt/app/src/
# Deterministic CoT artifacts (decision graph, roots, organ map, traversal engine)
COPY --chown=user:user cot_artifacts/ /opt/app/cot_artifacts/
# Vendored TRIDENT package (seg -> coords -> H-Optimus features)
COPY --chown=user:user trident/     /opt/app/trident/

ENTRYPOINT ["python", "inference.py"]
