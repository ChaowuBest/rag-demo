"""
LangChain 版 RAG（功能同 rag.py，含哈希增量同步）
================================================
安装：
  pip install langchain-classic langchain-chroma langchain-huggingface langchain-openai \
              langchain-text-splitters greenlet

运行：
  export LLM_API_KEY="你的key"
  python rag_langchain.py          # 同步 + 问答
  python rag_langchain.py sync     # 只同步

增量同步由 LangChain Indexing API 负责：
  - 每个文本块（内容 + metadata）算哈希，记录在 record_manager.db（SQLite）里
  - 没变的块跳过；变了的块重新编码；文件删掉后它的块也会被清掉（cleanup="full"）
  - 注意：chroma_db/ 和 record_manager.db 要一起删、一起留，否则两边记录对不上
"""

import glob
import os
import sys

from langchain_chroma import Chroma
from langchain_classic.indexes import SQLRecordManager
from langchain_core.documents import Document
from langchain_core.indexing import index
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ========== 配置 ==========
DOCS_DIR = os.getenv("DOCS_DIR", "docs")
DB_DIR = os.getenv("DB_DIR", "chroma_db_lc")
RECORD_DB = "sqlite:///" + os.getenv("RECORD_DB_PATH", "record_manager.db")
COLLECTION = "knowledge"
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")

LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

CHUNK_SIZE = 300
CHUNK_OVERLAP = 50
TOP_K = 3


def load_and_split():
    """第 1+2 步：读 docs/ 下的 .txt/.md，按段落/句子优先切块"""
    docs = []
    for path in sorted(glob.glob(os.path.join(DOCS_DIR, "*"))):
        if path.endswith((".txt", ".md")):
            with open(path, encoding="utf-8") as f:
                docs.append(Document(page_content=f.read(),
                                     metadata={"source": os.path.basename(path)}))
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    return splitter.split_documents(docs)


def get_store():
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL,
                                       encode_kwargs={"normalize_embeddings": True})
    store = Chroma(collection_name=COLLECTION, embedding_function=embeddings,
                   persist_directory=DB_DIR, collection_metadata={"hnsw:space": "cosine"})
    record_manager = SQLRecordManager(f"chroma/{COLLECTION}", db_url=RECORD_DB)
    record_manager.create_schema()
    return store, record_manager


def sync(store, record_manager):
    """第 3+4 步：哈希增量同步——这一个调用替代了 rag.py 里的 sync_index()"""
    chunks = load_and_split()
    if not chunks:
        raise SystemExit(f"在 {DOCS_DIR}/ 里没找到 .txt 或 .md 文件。")
    result = index(chunks, record_manager, store, cleanup="full", source_id_key="source",
                   key_encoder="sha256")
    print(f"同步完成：新增 {result['num_added']}，跳过 {result['num_skipped']}，"
          f"删除 {result['num_deleted']} 个文本块。")
    return result


PROMPT = (
    "请只根据下面的资料回答问题。如果资料里没有答案，就说“资料里没有提到”。\n\n"
    "资料：\n{context}\n\n问题：{question}"
)


def ask(question, store, llm):
    """第 5 步：检索 + 让大模型回答"""
    docs = store.similarity_search(question, k=TOP_K)
    context = "\n\n".join(f"【来自 {d.metadata['source']}】\n{d.page_content}" for d in docs)
    answer = llm.invoke([
        ("system", "你是一个根据知识库回答问题的助手。"),
        ("user", PROMPT.format(context=context, question=question)),
    ]).content
    return answer, sorted({d.metadata["source"] for d in docs})


def main():
    sync_only = sys.argv[1:] == ["sync"]
    if not sync_only and not LLM_API_KEY:
        raise SystemExit("请先设置环境变量 LLM_API_KEY（只同步可运行 python rag_langchain.py sync）")

    store, record_manager = get_store()
    sync(store, record_manager)
    if sync_only:
        return

    llm = ChatOpenAI(model=LLM_MODEL, api_key=LLM_API_KEY, base_url=LLM_BASE_URL)
    while True:
        question = input("\n你的问题（直接回车退出）：").strip()
        if not question:
            break
        answer, sources = ask(question, store, llm)
        print(f"\n回答：{answer}")
        print(f"参考文件：{', '.join(sources)}")


if __name__ == "__main__":
    main()
