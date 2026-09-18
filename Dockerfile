FROM python:3.12-slim

# 国内可用镜像源（构建时覆盖：docker build --build-arg PIP_INDEX=https://pypi.org/simple .）
ARG PIP_INDEX=https://pypi.tuna.tsinghua.edu.cn/simple

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_INDEX_URL=${PIP_INDEX} \
    HF_ENDPOINT=https://hf-mirror.com

# 依赖单独一层，改代码不必重装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY scripts ./scripts
COPY tests ./tests
COPY data ./data
COPY start.bat requirements-dev.txt README.md 部署说明.md ./

EXPOSE 8000

# 容器内需监听 0.0.0.0 才能被宿主机访问
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
