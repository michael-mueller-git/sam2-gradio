FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04

WORKDIR /app

RUN apt-get update \
    && apt-get install python3.10 python-is-python3 pip git curl wget ffmpeg -y

RUN mkdir -p /app/checkpoints

RUN wget -c https://huggingface.co/facebook/sam2-hiera-large/resolve/main/sam2_hiera_large.pt -O /app/checkpoints/sam2_hiera_large.pt
RUN wget -c https://huggingface.co/facebook/sam2-hiera-large/raw/main/sam2_hiera_l.yaml -O /app/sam2_hiera_l.yaml

COPY src/requirements.txt /app

RUN pip install -r requirements.txt
RUN pip install git+https://github.com/facebookresearch/segment-anything-2
# Download model
RUN python -c "from torchvision.models.detection import fasterrcnn_resnet50_fpn; fasterrcnn_resnet50_fpn(pretrained=True)"

RUN mkdir -p /app/images
RUN mkdir -p /app/output

COPY src/ /app
COPY entrypoint.sh /app
RUN chmod +x /app/entrypoint.sh

ENTRYPOINT ["/app/entrypoint.sh"]
