"""
最简单的 RAG 知识库
==================
流程：
  1. 读取 docs/ 文件夹里的文件（.txt / .md）
  2. 把每个文件切成小块
  3. 用 embedding 模型把每块文字变成向量
  4. 存进 Chroma 向量数据库（保存在 chroma_db/ 文件夹）
  5. 你提问 → 找出最相关的几块 → 连同问题一起发给大模型 → 大模型回答

安装：
  pip install chromadb sentence-transformers openai

运行：
  export LLM_API_KEY="你的key"
  python rag.py

说明：
  - embedding 模型在本地跑，第一次运行会自动下载（约 100MB）
  - 大模型用 OpenAI 兼容接口，默认 DeepSeek；换别家只要改下面三个配置
  - 每次启动会按文件哈希自动同步 docs/：新增/修改/删除的文件会更新，没变的跳过
  - 只同步不提问：python rag.py sync
  - 换了 embedding 模型或改了切块参数，要删掉 chroma_db/ 全量重建
"""

import os
import sys
import glob
import hashlib

import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ========== 配置 ==========
DOCS_DIR = os.getenv("DOCS_DIR", "docs")                       # 放你那 10 个文件的文件夹
DB_DIR = os.getenv("DB_DIR", "chroma_db")                    # 向量数据库存放位置
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-zh-v1.5")  # 中文 embedding 模型（英文文档可换 BAAI/bge-small-en-v1.5）

# 大模型（任何 OpenAI 兼容接口都行）
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "https://api.deepseek.com")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-flash")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
# 其他例子：
#   本地 Ollama: LLM_BASE_URL=http://localhost:11434/v1  LLM_MODEL=qwen2.5:7b  LLM_API_KEY=ollama
#   OpenAI:      LLM_BASE_URL=https://api.openai.com/v1  LLM_MODEL=你要用的模型名

CHUNK_SIZE = 300      # 每块多少个字符
CHUNK_OVERLAP = 50    # 相邻两块重叠多少字符（防止一句话被切断）
TOP_K = 3             # 每次检索几块给大模型


# ========== 第 1 步：读文件 ==========
def load_files(folder):
    files = []
    for path in sorted(glob.glob(os.path.join(folder, "*"))):
        if path.endswith((".txt", ".md")):
            with open(path, encoding="utf-8") as f:
                files.append((os.path.basename(path), f.read()))
    return files


# ========== 第 2 步：切块 ==========
def split_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start:start + size].strip()
        if chunk:
            chunks.append(chunk)
        start += size - overlap
    return chunks


# ========== 第 3+4 步：向量化并存入数据库（按文件哈希增量同步） ==========
def file_hash(content):
    """文件内容的指纹：内容不变指纹就不变，改一个字指纹就完全不同"""
    return hashlib.md5(content.encode("utf-8")).hexdigest()


def add_file(embedder, collection, name, content, h):
    """把一个文件切块、编码、写入数据库；每块都记下所属文件和文件哈希"""
    chunks = split_text(content)
    if not chunks:  # 空文件：没东西可存
        return 0
    vectors = embedder.encode(chunks, normalize_embeddings=True).tolist()
    collection.add(
        ids=[f"{name}#{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=vectors,
        metadatas=[{"source": name, "hash": h} for _ in chunks],
    )
    return len(chunks)


def sync_index(embedder, collection):
    """
    让数据库和 docs/ 保持一致，只处理有变化的文件：
      新文件         → 加入
      内容变了       → 先删旧块，再加入新块（直接覆盖会残留多出来的旧块）
      文件被删了     → 删掉它的所有块
      没变           → 跳过，不重新算 embedding
    返回 {"added": [...], "updated": [...], "deleted": [...], "unchanged": [...]}
    """
    files = load_files(DOCS_DIR)
    if not files:
        # 防止在错误的目录下运行把整个库清空
        raise SystemExit(f"在 {DOCS_DIR}/ 里没找到 .txt 或 .md 文件，先把文件放进去。")

    # 磁盘上：{文件名: (哈希, 内容)}
    disk = {name: (file_hash(content), content) for name, content in files}

    # 数据库里：{文件名: 哈希}（从每个块的 metadata 里读；老库没有 hash 字段会读成 None，当作已修改重建）
    stored = {}
    for m in collection.get(include=["metadatas"])["metadatas"]:
        stored[m["source"]] = m.get("hash")

    report = {"added": [], "updated": [], "deleted": [], "unchanged": []}

    # 1) 库里有，但磁盘上没了 → 删除
    for name in stored:
        if name not in disk:
            collection.delete(where={"source": name})
            report["deleted"].append(name)

    # 2) 磁盘上的每个文件：比对哈希
    for name, (h, content) in disk.items():
        if name not in stored:
            add_file(embedder, collection, name, content, h)
            report["added"].append(name)
        elif stored[name] != h:
            collection.delete(where={"source": name})
            add_file(embedder, collection, name, content, h)
            report["updated"].append(name)
        else:
            report["unchanged"].append(name)

    labels = {"added": "新增", "updated": "更新", "deleted": "删除", "unchanged": "未变"}
    print("同步完成：" + "，".join(f"{labels[k]} {len(v)}" for k, v in report.items())
          + f"，库里共 {collection.count()} 个文本块。")
    for k in ("added", "updated", "deleted"):
        for name in report[k]:
            print(f"  [{labels[k]}] {name}")
    return report


# ========== 第 5 步：检索 + 让大模型回答 ==========
def ask(question, embedder, collection, llm):
    # 5a. 把问题也变成向量，在数据库里找最相近的几块
    q_vec = embedder.encode([question], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=q_vec, n_results=TOP_K)
    chunks = result["documents"][0]
    sources = [m["source"] for m in result["metadatas"][0]]

    # 5b. 把找到的内容拼进提示词
    context = "\n\n".join(f"【来自 {s}】\n{c}" for s, c in zip(sources, chunks))
    prompt = (
        "请只根据下面的资料回答问题。如果资料里没有答案，就说“资料里没有提到”。\n\n"
        f"资料：\n{context}\n\n"
        f"问题：{question}"
    )

    # 5c. 发给大模型
    response = llm.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是一个根据知识库回答问题的助手。"},
            {"role": "user", "content": prompt},
        ],
    )
    return response.choices[0].message.content, sorted(set(sources))


def main():
    # python rag.py sync  → 只同步知识库，不需要 LLM key
    sync_only = sys.argv[1:] == ["sync"]
    if not sync_only and not LLM_API_KEY:
        raise SystemExit("请先设置环境变量 LLM_API_KEY（只想同步知识库可以运行 python rag.py sync）")

    print("加载 embedding 模型...")
    embedder = SentenceTransformer(EMBED_MODEL)

    db = chromadb.PersistentClient(path=DB_DIR)
    collection = db.get_or_create_collection("knowledge", metadata={"hnsw:space": "cosine"})

    # 每次启动都和 docs/ 对比一次哈希，只处理变化的文件
    sync_index(embedder, collection)
    if sync_only:
        return

    llm = OpenAI(api_key=LLM_API_KEY, base_url=LLM_BASE_URL)

    while True:
        question = input("\n你的问题（直接回车退出）：").strip()
        if not question:
            break
        answer, sources = ask(question, embedder, collection, llm)
        print(f"\n回答：{answer}")
        print(f"参考文件：{', '.join(sources)}")


if __name__ == "__main__":
    main()
