# JUMP.LAB proof-of-concept, CPU-only MMPose. Build and memory usage may exceed Render Free.
FROM python:3.10-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MMPOSE_DEVICE=cpu MMPOSE_CONFIG=/opt/mmpose/configs/wholebody_2d_keypoint/rtmpose/coco-wholebody/rtmpose-m_8xb64-270e_coco-wholebody-256x192.py MMPOSE_CHECKPOINT=/app/models/rtmpose-m-coco-wholebody.pth
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git ffmpeg libglib2.0-0 libgl1 && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade 'pip<25' 'setuptools<70' wheel 'cython<3' && python -m pip install 'numpy==1.26.4'
RUN python -m pip install 'torch==2.1.0+cpu' 'torchvision==0.16.0+cpu' --index-url https://download.pytorch.org/whl/cpu
RUN python -m pip install 'mmcv==2.1.0' --no-deps -f https://download.openmmlab.com/mmcv/dist/cpu/torch2.1.0/index.html && python -m pip install 'mmengine==0.10.7'
RUN git clone --depth 1 --branch v1.3.2 https://github.com/open-mmlab/mmpose.git /opt/mmpose && python -m pip install --no-build-isolation -e /opt/mmpose
COPY requirements.txt /app/requirements.txt
RUN python -m pip install -r requirements.txt && python -m pip install --force-reinstall --no-deps 'numpy==1.26.4'
RUN mkdir -p /app/models && curl -fLsS --retry 3 --retry-delay 3 'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/rtmpose-m_simcc-coco-wholebody_pt-aic-coco_270e-256x192-cd5e845c_20230123.pth' -o /app/models/rtmpose-m-coco-wholebody.pth && test -s /app/models/rtmpose-m-coco-wholebody.pth
COPY main.py ./main.py
EXPOSE 10000
CMD ["sh","-c","exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
