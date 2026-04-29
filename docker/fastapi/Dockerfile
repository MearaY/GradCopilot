# 使用官方 Python 3.13 精简版基础镜像
FROM python:3.13-slim

# 设置环境变量，防止 Python 缓冲 stdout 和生成 pyc 文件
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# 设置工作目录
WORKDIR /app

# 安装必要的系统级依赖（如编译扩展、连接 PG 所需）
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    python3-dev \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖定义文件
COPY requirements.txt /app/

# 安装 Python 依赖，禁止缓存以减小镜像体积
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目源代码
COPY src/ /app/src/
COPY migrations/ /app/migrations/

# 暴露 FastAPI 默认端口
EXPOSE 8000

# 启动命令（无需在此运行迁移，迁移由 docker-compose 或用户手动执行）
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
