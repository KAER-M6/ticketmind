FROM python:3.12-slim

# 国内可用镜像源（构建时覆盖：docker build --build-arg PIP_INDEX=https://pypi.org/simple .）
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple
# CPU 版 torch 专用源：本项目只用 CPU 推理，必须显式指定
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_INDEX_URL=${PIP_INDEX} \
    HF_ENDPOINT=https://hf-mirror.com

# 依赖单独一层，改代码不必重装依赖
COPY requirements.txt .

# 先单独装 CPU 版 torch。
# 关键：Linux 上 pip install torch 默认解析到 CUDA 版（体积 2GB+，还会拖入 nvidia-* 系列依赖），
# 而本项目只做 CPU 推理。先装 CPU 版，下一步安装 requirements.txt 时该依赖即已满足。
# 如源不可达，可用 --build-arg TORCH_INDEX=... 换成镜像站。
RUN pip install --no-cache-dir --index-url ${TORCH_INDEX} torch

RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY tests ./tests
COPY data ./data
COPY README.md ./

EXPOSE 8000

# 容器内需监听 0.0.0.0 才能被宿主机访问
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
