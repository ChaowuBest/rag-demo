# rag-demo

两个版本的最简 RAG 知识库（功能相同）：

- `rag-plain/`：纯 Python（chromadb + sentence-transformers），按文件哈希增量同步
- `rag-langchain/`：LangChain Indexing API，按文本块哈希增量同步

## Docker（推荐）

镜像已内置 embedding 模型（BAAI/bge-small-zh-v1.5）和 CPU 版 torch，支持 amd64 / arm64，运行时不需要访问 Hugging Face。

| 镜像 | 说明 |
|---|---|
| `ghcr.io/chaowubest/rag-demo:plain` | 纯 Python 版 |
| `ghcr.io/chaowubest/rag-demo:langchain` | LangChain 版 |

固定版本：`:0.1.0-plain` / `:0.1.0-langchain`

```bash
# 问答（-it 才能输入问题）
docker run --rm -it \
  -e LLM_API_KEY=你的key -e LLM_MODEL=deepseek-chat \
  -v "$PWD/docs":/app/docs:ro \
  -v rag-data:/data \
  ghcr.io/chaowubest/rag-demo:plain

# 只同步知识库（不需要 key）
docker run --rm -v "$PWD/docs":/app/docs:ro -v rag-data:/data ghcr.io/chaowubest/rag-demo:plain sync
```

- `-v .../docs:/app/docs`：你的文档；不挂载就用镜像里自带的 4 个示例文档
- `-v rag-data:/data`：向量库存这里，下次运行只处理变化的文件；两个版本请用不同的卷
- 换成其他 OpenAI 兼容接口：`-e LLM_BASE_URL=... -e LLM_MODEL=...`
- 容器里访问本机 Ollama：`-e LLM_BASE_URL=http://host.docker.internal:11434/v1 -e LLM_MODEL=qwen2.5:7b -e LLM_API_KEY=ollama`

## 自己构建

```bash
docker build --target plain     -t rag-demo:plain .
docker build --target langchain -t rag-demo:langchain .
```

## 不用 Docker

见各文件夹里的 README.md。
