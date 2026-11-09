"""自動測試：確認「切段」和「檢索」的行為符合預期。

執行方式（在專案根目錄）：
    python -m unittest discover -s tests -t . -v

這些測試只測本機的切段與檢索，不呼叫 Claude API，不花錢。
第一次執行會下載 embedding 模型（約 0.2 GB）。
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
    """檢索標準：正確的來源要出現在前 3 名內。

    原因：ask.py 會把前 3 段一起交給 Claude，所以只要正確的那篇在前 3 名，
    Claude 就看得到答案。
    """

    @classmethod
    def setUpClass(cls):
        # 用暫存的索引檔測試，不動到使用者自己的 index.pkl
        cls._tmp = tempfile.TemporaryDirectory()
        chunks = ingest.load_chunks(SAMPLE_DIR)
        matrix = ingest.build_index(chunks)
        index_path = Path(cls._tmp.name) / "index.pkl"
        with open(index_path, "wb") as f:
            pickle.dump({"matrix": matrix, "chunks": chunks, "model": ingest.MODEL_NAME}, f)
        cls._original = ask.INDEX_FILE
        ask.INDEX_FILE = index_path

    @classmethod
    def tearDownClass(cls):
        ask.INDEX_FILE = cls._original
        cls._tmp.cleanup()

    def sources_in_top3(self, question):
        return [h["source"] for h in ask.retrieve(question)]

    def test_top_k_returns_three_hits_with_scores(self):
        hits = ask.retrieve("爬蟲被 403 擋掉怎麼辦")
        self.assertEqual(len(hits), 3)
        self.assertGreaterEqual(hits[0]["score"], hits[1]["score"])

    def test_literal_overlap(self):
        self.assertIn("2026-10-05.md", self.sources_in_top3("爬蟲被 403 擋掉怎麼辦"))

    def test_synonym_phrase(self):
        # 日誌寫「403 被擋」，問題說「不讓我進去」→ 舊版 TF-IDF 失敗，embedding 應該成功
        hits = self.sources_in_top3("網站不讓我進去怎麼辦")
        self.assertEqual(hits[0], "2026-10-05.md")

    def test_cross_language(self):
        # 英文問中文日誌 → 舊版 TF-IDF 失敗，embedding 應該成功
        hits = self.sources_in_top3("what did I learn about loops")
        self.assertEqual(hits[0], "2026-10-01.md")

    def test_mcp_question(self):
        self.assertIn("2026-10-12.md", self.sources_in_top3("我怎麼讓AI接上外部工具"))

    def test_job_search_question(self):
        self.assertIn("2026-10-20.md", self.sources_in_top3("我找工作準備得怎麼樣"))


if __name__ == "__main__":
    unittest.main()
