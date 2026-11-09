"""自動測試：確認「切段」和「檢索」的行為符合預期。

執行方式（在專案根目錄）：
    python -m unittest discover -s tests -t . -v

這些測試只測本機的切段與檢索，不呼叫 Claude API，不花錢。
"""

import pickle
import tempfile
import unittest
from pathlib import Path

import ask
import ingest

SAMPLE_DIR = Path(__file__).parent.parent / "sample_data"


class TestChunking(unittest.TestCase):
    def test_four_journals_become_twelve_chunks(self):
        chunks = ingest.load_chunks(SAMPLE_DIR)
        self.assertEqual(len(chunks), 12)  # 4 篇 × 3 個標題

    def test_each_chunk_keeps_source_and_heading(self):
        for c in ingest.load_chunks(SAMPLE_DIR):
            self.assertTrue(c["source"].endswith(".md"))
            self.assertIn(c["heading"], ["今天學了什麼", "遇到的困難", "心得"])
            self.assertTrue(c["text"].strip())


class TestRetrieval(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # 用暫存的索引檔測試，不動到使用者自己的 index.pkl
        cls._tmp = tempfile.TemporaryDirectory()
        chunks = ingest.load_chunks(SAMPLE_DIR)
        vectorizer, matrix = ingest.build_index(chunks)
        index_path = Path(cls._tmp.name) / "index.pkl"
        with open(index_path, "wb") as f:
            pickle.dump({"vectorizer": vectorizer, "matrix": matrix, "chunks": chunks}, f)
        cls._original = ask.INDEX_FILE
        ask.INDEX_FILE = index_path

    @classmethod
    def tearDownClass(cls):
        ask.INDEX_FILE = cls._original
        cls._tmp.cleanup()

    def top_source(self, question):
        hits = ask.retrieve(question)
        return hits[0]["source"] if hits else None

    # ---- 預期會成功的情況：問題和日誌有共同用字 ----
    def test_crawler_question_finds_crawler_journal(self):
        self.assertEqual(self.top_source("爬蟲被 403 擋掉怎麼辦"), "2026-10-05.md")

    def test_mcp_question_finds_mcp_journal(self):
        self.assertEqual(self.top_source("我怎麼讓AI接上外部工具"), "2026-10-12.md")

    def test_job_search_question_finds_resume_journal(self):
        self.assertEqual(self.top_source("我找工作準備得怎麼樣"), "2026-10-20.md")

    # ---- 已知限制：TF-IDF 只比對字面，不懂同義說法 ----
    @unittest.expectedFailure
    def test_known_limit_paraphrase(self):
        # 日誌寫「403 被擋」，問題說「不讓我進去」→ 字面沒重疊，找不到
        self.assertEqual(self.top_source("網站不讓我進去怎麼辦"), "2026-10-05.md")

    @unittest.expectedFailure
    def test_known_limit_cross_language(self):
        # 英文問中文日誌 → 完全沒有共同用字
        self.assertEqual(self.top_source("what did I learn about loops"), "2026-10-01.md")


if __name__ == "__main__":
    unittest.main()
