FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04
WORKDIR /content
ENV PATH="/home/camenduru/.local/bin:${PATH}"
RUN adduser --disabled-password --gecos '' camenduru && \
    adduser camenduru sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && \
    chown -R camenduru:camenduru /content && \
    chmod -R 777 /content && \
    chown -R camenduru:camenduru /home && \
    chmod -R 777 /home

RUN apt update -y && add-apt-repository -y ppa:git-core/ppa && apt update -y && apt install -y aria2 git git-lfs unzip ffmpeg

USER camenduru

RUN pip install -q opencv-python imageio imageio-ffmpeg ffmpeg-python av runpod \
    xformers==0.0.25 torchsde==0.2.6 einops==0.8.0 diffusers==0.28.0 transformers==4.41.2 accelerate==0.30.1 pyyaml numpy==1.26.4 onnxruntime-gpu pykalman mediapipe onnx2torch \
    pillow==10.3.0 scipy color-matcher matplotlib huggingface_hub mss

RUN git clone -b liveportrait https://github.com/camenduru/ComfyUI /content/ComfyUI && \
    git clone -b tost https://github.com/camenduru/ComfyUI-LivePortraitKJ /content/ComfyUI/custom_nodes/ComfyUI-LivePortraitKJ && \
    git clone -b tost https://github.com/camenduru/ComfyUI-KJNodes /content/ComfyUI/custom_nodes/ComfyUI-KJNodes && \
    git clone -b tost https://github.com/camenduru/ComfyUI-VideoHelperSuite /content/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite

RUN aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/insightface/buffalo_l/1k3d68.onnx -d /content/ComfyUI/models/insightface/models/buffalo_l -o 1k3d68.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/insightface/buffalo_l/2d106det.onnx -d /content/ComfyUI/models/insightface/models/buffalo_l -o 2d106det.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/insightface/buffalo_l/det_10g.onnx -d /content/ComfyUI/models/insightface/models/buffalo_l -o det_10g.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/insightface/buffalo_l/genderage.onnx -d /content/ComfyUI/models/insightface/models/buffalo_l -o genderage.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/insightface/buffalo_l/w600k_r50.onnx -d /content/ComfyUI/models/insightface/models/buffalo_l -o w600k_r50.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/liveportrait/human/appearance_feature_extractor.safetensors -d /content/ComfyUI/models/liveportrait -o appearance_feature_extractor.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/liveportrait/human/landmark.onnx -d /content/ComfyUI/models/liveportrait -o landmark.onnx && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/liveportrait/human/landmark_model.pth -d /content/ComfyUI/models/liveportrait -o landmark_model.pth && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/liveportrait/human/motion_extractor.safetensors -d /content/ComfyUI/models/liveportrait -o motion_extractor.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/liveportrait/human/spade_generator.safetensors -d /content/ComfyUI/models/liveportrait -o spade_generator.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/liveportrait/human/stitching_retargeting_module.safetensors -d /content/ComfyUI/models/liveportrait -o stitching_retargeting_module.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/liveportrait/human/warping_module.safetensors -d /content/ComfyUI/models/liveportrait -o warping_module.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/liveportrait/animal/appearance_feature_extractor.safetensors -d /content/ComfyUI/models/liveportrait/animal -o appearance_feature_extractor.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/liveportrait/animal/motion_extractor.safetensors -d /content/ComfyUI/models/liveportrait/animal -o motion_extractor.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/liveportrait/animal/spade_generator.safetensors -d /content/ComfyUI/models/liveportrait/animal -o spade_generator.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/liveportrait/animal/stitching_retargeting_module.safetensors -d /content/ComfyUI/models/liveportrait/animal -o stitching_retargeting_module.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M  https://huggingface.co/camenduru/LivePortrait_InsightFace/resolve/main/liveportrait/animal/warping_module.safetensors -d /content/ComfyUI/models/liveportrait/animal -o warping_module.safetensors

COPY ./worker_runpod.py /content/ComfyUI/worker_runpod.py
WORKDIR /content/ComfyUI
CMD python worker_runpod.py