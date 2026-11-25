"""
测试按哈希增量同步：python -m unittest test_sync -v
用假的 embedder（不加载模型），几秒内跑完；记录每次被编码的文本，用来确认"没变的文件不会重新编码"。
"""

import os
import tempfile
import unittest
import uuid

import chromadb

import rag


class FakeEmbedder:
    def __init__(self):
        self.encoded = []  # 每次 encode 收到的文本

    def encode(self, texts, normalize_embeddings=True):
        import numpy as np
        self.encoded.extend(texts)
        return np.array([[float(len(t)), 1.0, 0.0] for t in texts])


class SyncIndexTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old_docs = rag.DOCS_DIR
        rag.DOCS_DIR = self.tmp.name
        self.collection = chromadb.EphemeralClient().get_or_create_collection(
            f"t{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"}
        )
        self.emb = FakeEmbedder()

    def tearDown(self):
        rag.DOCS_DIR = self.old_docs
        self.tmp.cleanup()

    def write(self, name, text):
        with open(os.path.join(self.tmp.name, name), "w", encoding="utf-8") as f:
            f.write(text)

    def remove(self, name):
        os.remove(os.path.join(self.tmp.name, name))

    def sync(self):
        self.emb.encoded.clear()
        return rag.sync_index(self.emb, self.collection)

    def sources(self):
        return sorted({m["source"] for m in self.collection.get()["metadatas"]})

    def docs_of(self, name):
        return self.collection.get(where={"source": name})["documents"]

    def test_first_run_adds_everything(self):
        self.write("a.md", "年假15天")
        self.write("b.txt", "报销30天内提交")
        r = self.sync()
        self.assertEqual(sorted(r["added"]), ["a.md", "b.txt"])
        self.assertEqual(self.sources(), ["a.md", "b.txt"])

    def test_second_run_unchanged_skips_encoding(self):
        self.write("a.md", "年假15天")
        self.sync()
        r = self.sync()
        self.assertEqual(r["unchanged"], ["a.md"])
        self.assertEqual(self.emb.encoded, [], "没变的文件不应该重新编码")

    def test_modify_delete_add(self):
        self.write("假期.md", "年假15天")
        self.write("报销.md", "报销30天内提交")
        self.write("IT.txt", "IT分机8888")
        self.sync()

        self.write("假期.md", "年假18天")   # 修改
        self.remove("IT.txt")               # 删除
        self.write("考勤.md", "9点上班")     # 新增
        r = self.sync()

        self.assertEqual(r["updated"], ["假期.md"])
        self.assertEqual(r["deleted"], ["IT.txt"])
        self.assertEqual(r["added"], ["考勤.md"])
        self.assertEqual(r["unchanged"], ["报销.md"])
        self.assertEqual(self.sources(), ["假期.md", "报销.md", "考勤.md"])
        self.assertEqual(self.docs_of("假期.md"), ["年假18天"])
        # 只有修改和新增的文件被编码，报销.md 没有
        self.assertEqual(sorted(self.emb.encoded), sorted(["年假18天", "9点上班"]))

    def test_shrinking_file_leaves_no_stale_chunks(self):
        self.write("长文.md", "甲" * 900)   # 切成多块
        self.sync()
        before = len(self.docs_of("长文.md"))
        self.write("长文.md", "乙" * 100)   # 只剩 1 块
        self.sync()
        docs = self.docs_of("长文.md")
        self.assertGreater(before, 1)
        self.assertEqual(docs, ["乙" * 100], "旧的多余块必须被删掉")

    def test_old_db_without_hash_gets_rebuilt(self):
        # 升级前建的库：metadata 里没有 hash 字段
        self.write("a.md", "年假15天")
        self.collection.add(ids=["a.md#0"], documents=["年假15天"],
                            embeddings=[[1.0, 1.0, 0.0]], metadatas=[{"source": "a.md"}])
        r = self.sync()
        self.assertEqual(r["updated"], ["a.md"])
        self.assertEqual(self.collection.get()["metadatas"][0]["hash"], rag.file_hash("年假15天"))

    def test_empty_docs_dir_does_not_wipe_db(self):
        self.write("a.md", "年假15天")
        self.sync()
        self.remove("a.md")
        with self.assertRaises(SystemExit):
            self.sync()
        self.assertEqual(self.sources(), ["a.md"])


if __name__ == "__main__":
    unittest.main()
