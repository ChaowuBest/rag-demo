# RAG 知识库（纯 Python 版）

chromadb + sentence-transformers（BAAI/bge-small-zh-v1.5）+ OpenAI 兼容接口。
每次启动按**文件哈希**增量同步 `docs/`：新增/修改/删除的文件会更新，没变的跳过。

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python rag.py sync                                   # 只同步知识库
LLM_API_KEY=xxx LLM_MODEL=deepseek-chat python rag.py  # 同步 + 问答

python -m unittest test_sync -v    # 增量同步的单元测试（不加载模型，秒级）
python test_embedding.py           # embedding 模型效果测试（会下载约 100MB 模型）
```

- 把 `docs/` 里的示例文件换成你自己的 `.txt` / `.md`
- 换了 embedding 模型或改了切块参数：删掉 `chroma_db/` 全量重建
