# 两个版本共用一个 Dockerfile：
#   docker build --target plain     -t rag-demo:plain .
#   docker build --target langchain -t rag-demo:langchain .

# ---------- 公共层：CPU 版 torch + sentence-transformers + 预下载 embedding 模型 ----------
FROM python:3.13-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models
ARG EMBED_MODEL=BAAI/bge-small-zh-v1.5
# 先装 CPU 版 torch，避免 amd64 上装成几个 GB 的 CUDA 版
RUN pip install "torch==2.15.*" --index-url https://download.pytorch.org/whl/cpu \
 && pip install "sentence-transformers>=6.3,<7"
# 模型打进镜像，运行时不联网也能用
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBED_MODEL}')" \
 && chmod -R a+rX /models
RUN useradd -m -u 1000 app && mkdir -p /data && chown app /data
WORKDIR /app
ENV HF_HUB_OFFLINE=1 \
    EMBED_MODEL=${EMBED_MODEL} \
    DOCS_DIR=/app/docs \
    LLM_BASE_URL=https://api.deepseek.com
VOLUME /data

# ---------- 纯 Python 版 ----------
FROM base AS plain
COPY rag-plain/requirements.txt .
RUN pip install -r requirements.txt
COPY rag-plain/ .
ENV DB_DIR=/data/chroma_db
USER app
ENTRYPOINT ["python", "rag.py"]

# ---------- LangChain 版 ----------
FROM base AS langchain
COPY rag-langchain/requirements.txt .
RUN pip install -r requirements.txt
COPY rag-langchain/ .
ENV DB_DIR=/data/chroma_db_lc \
    RECORD_DB_PATH=/data/record_manager.db
USER app
ENTRYPOINT ["python", "rag_langchain.py"]
