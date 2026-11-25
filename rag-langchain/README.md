# RAG 知识库（LangChain 版）

功能同纯 Python 版，增量同步改用 LangChain Indexing API（`index()` + `SQLRecordManager`），
按**文本块哈希**判断变化；切块用 `RecursiveCharacterTextSplitter`（优先在段落、句号处切）。

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python rag_langchain.py sync                       # 只同步知识库
LLM_API_KEY=xxx python rag_langchain.py            # 同步 + 问答（默认 deepseek-chat）
```

- 哈希记录在 `record_manager.db`，向量在 `chroma_db_lc/`：**两者要一起删、一起留**
- 换了 embedding 模型或改了切块参数：两者都删掉，全量重建
