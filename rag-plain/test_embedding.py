"""
测试 embedding 模型：加载 → 看向量维度 → 相似度是否合理 → 用 docs/ 做检索测试
运行：.venv/bin/python test_embedding.py
"""

import time

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

from rag import EMBED_MODEL, sync_index, split_text, load_files, DOCS_DIR, TOP_K

# bge 中文模型建议：短问题检索长段落时，给问题加这个前缀（文档不加）
QUERY_PREFIX = "为这个句子生成表示以用于检索相关文章："


def section(title):
    print(f"\n{'=' * 10} {title} {'=' * 10}")


# ---------- 1. 加载 ----------
section("1. 加载模型")
t = time.time()
model = SentenceTransformer(EMBED_MODEL)
print(f"模型: {EMBED_MODEL}")
print(f"加载耗时: {time.time() - t:.1f}s  设备: {model.device}")
print(f"向量维度: {model.get_embedding_dimension()}  最大长度: {model.max_seq_length} tokens")

# ---------- 2. 基本编码 ----------
section("2. 编码")
vec = model.encode(["今天天气真好"], normalize_embeddings=True)
print(f"shape={vec.shape}  范数={np.linalg.norm(vec[0]):.4f}（归一化后应为 1）")
assert vec.shape == (1, model.get_embedding_dimension())

# ---------- 3. 语义相似度 ----------
section("3. 语义相似度")
pairs = [
    ("我想请几天假", "年假怎么申请", "相关"),
    ("出差住酒店能报多少钱", "住宿标准每晚不超过600元", "相关"),
    ("电脑密码忘了", "忘记开机密码请联系IT服务台", "相关"),
    ("我想请几天假", "专业版每月18元", "无关"),
    ("今天天气真好", "VPN需要绑定手机验证器", "无关"),
]
a = model.encode([p[0] for p in pairs], normalize_embeddings=True)
b = model.encode([p[1] for p in pairs], normalize_embeddings=True)
sims = (a * b).sum(axis=1)
for (x, y, label), s in zip(pairs, sims):
    print(f"  {s:.3f}  [{label}] {x}  <->  {y}")
related = [s for (_, _, l), s in zip(pairs, sims) if l == "相关"]
unrelated = [s for (_, _, l), s in zip(pairs, sims) if l == "无关"]
assert min(related) > max(unrelated), "相关句子的相似度应该都高于无关句子"
print(f"OK：相关最低 {min(related):.3f} > 无关最高 {max(unrelated):.3f}")

# ---------- 4. 检索测试（内存版 Chroma，不影响 chroma_db/） ----------
section("4. 检索测试")
collection = chromadb.EphemeralClient().get_or_create_collection(
    "test", metadata={"hnsw:space": "cosine"}
)
sync_index(model, collection)

cases = [
    ("年假有几天？", "公司假期制度.md"),
    ("病假要医院证明吗", "公司假期制度.md"),
    ("出差住宿能报销多少", "报销流程.md"),
    ("报销的钱什么时候到账", "报销流程.md"),
    ("专业版多少钱一个月", "产品介绍.txt"),
    ("VPN 怎么登录", "IT支持.txt"),
    ("访客用哪个 WiFi", "IT支持.txt"),
]


def run(prefix):
    hits = 0
    for q, expected in cases:
        qv = model.encode([prefix + q], normalize_embeddings=True).tolist()
        r = collection.query(query_embeddings=qv, n_results=TOP_K)
        top1 = r["metadatas"][0][0]["source"]
        dist = r["distances"][0][0]
        ok = top1 == expected
        hits += ok
        print(f"  {'✓' if ok else '✗'} {q:<16} → {top1:<14} (相似度 {1 - dist:.3f})")
    return hits


for name, prefix in [("不加前缀", ""), ("加 bge 检索前缀", QUERY_PREFIX)]:
    print(f"\n[{name}]")
    hits = run(prefix)
    print(f"  Top-1 命中率: {hits}/{len(cases)}")

# ---------- 5. 速度 ----------
section("5. 速度")
texts = [c for _, content in load_files(DOCS_DIR) for c in split_text(content)] * 50
t = time.time()
model.encode(texts, normalize_embeddings=True, batch_size=32)
dt = time.time() - t
print(f"{len(texts)} 个文本块耗时 {dt:.2f}s（{len(texts) / dt:.0f} 块/秒）")
