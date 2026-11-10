"""MCP server 的端到端測試：真的啟動 server，用 MCP client 連線並呼叫工具。

執行方式（在專案根目錄）：
    python -m unittest discover -s tests -t . -v

這些測試不呼叫 Claude API，不花錢。
"""

import os
import pickle
import sys
import tempfile
import unittest
from pathlib import Path

from mcp import Client
from mcp.client.stdio import StdioServerParameters

import ingest

ROOT = Path(__file__).parent.parent
SAMPLE_DIR = ROOT / "sample_data"


class TestMCPServer(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # 用暫存索引 + 環境變數 JOURNAL_INDEX，讓 server 讀假資料，不受使用者自己的索引影響
        cls._tmp = tempfile.TemporaryDirectory()
        chunks = ingest.load_chunks(SAMPLE_DIR)
        index_path = Path(cls._tmp.name) / "index.pkl"
        with open(index_path, "wb") as f:
            pickle.dump(
                {"matrix": ingest.build_index(chunks), "chunks": chunks, "model": ingest.MODEL_NAME}, f
            )
        cls.params = StdioServerParameters(
            command=sys.executable,
            args=[str(ROOT / "mcp_server.py")],
            env={**os.environ, "JOURNAL_INDEX": str(index_path)},
            cwd=str(ROOT),
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    async def test_server_advertises_search_journal_tool(self):
        async with Client(self.params) as client:
            result = await client.list_tools()
            names = [t.name for t in result.tools]
            self.assertIn("search_journal", names)
            tool = next(t for t in result.tools if t.name == "search_journal")
            self.assertIn("日誌", tool.description)  # Claude 靠說明判斷何時使用

    async def test_search_returns_relevant_journal(self):
        async with Client(self.params) as client:
            result = await client.call_tool("search_journal", {"question": "網站不讓我進去怎麼辦"})
            text = result.content[0].text
            self.assertIn("2026-10-05.md", text)
            self.assertIn("403", text)

    async def test_top_k_is_respected(self):
        async with Client(self.params) as client:
            result = await client.call_tool(
                "search_journal", {"question": "Python 迴圈", "top_k": 1}
            )
            self.assertEqual(result.content[0].text.count("【"), 1)


    async def test_server_advertises_all_three_tools(self):
        async with Client(self.params) as client:
            names = {t.name for t in (await client.list_tools()).tools}
            self.assertEqual(names, {"search_journal", "list_journals", "read_journal"})

    async def test_list_journals(self):
        async with Client(self.params) as client:
            result = await client.call_tool("list_journals", {})
            names = result.content[0].text.splitlines()
            self.assertEqual(len(names), 4)
            self.assertIn("2026-10-05.md", names)

    async def test_read_journal_returns_content(self):
        async with Client(self.params) as client:
            result = await client.call_tool("read_journal", {"filename": "2026-10-05.md"})
            self.assertFalse(result.is_error)
            self.assertIn("403", result.content[0].text)

    async def test_read_journal_blocks_path_traversal(self):
        # 在日誌資料夾「外面」放一個機密檔，確認透過 MCP 也讀不到它
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "secret.txt").write_text("SECRET=abc123", encoding="utf-8")
            journals = Path(tmp) / "journals"
            journals.mkdir()
            (journals / "2026-01-01.md").write_text("# 一般日誌", encoding="utf-8")
            params = StdioServerParameters(
                command=sys.executable,
                args=[str(ROOT / "mcp_server.py")],
                env={**os.environ, "JOURNAL_DIR": str(journals)},
                cwd=str(ROOT),
            )
            async with Client(params) as client:
                for evil in ["../secret.txt", "../../etc/passwd", "/etc/passwd", "", "secret.txt"]:
                    result = await client.call_tool("read_journal", {"filename": evil})
                    self.assertTrue(result.is_error, evil)
                    self.assertNotIn("SECRET", result.content[0].text, evil)
                    # 錯誤訊息要真的傳到 Claude，它才知道該改用 list_journals
                    self.assertIn("list_journals", result.content[0].text, evil)
                ok = await client.call_tool("read_journal", {"filename": "2026-01-01.md"})
                self.assertIn("一般日誌", ok.content[0].text)  # 正常檔案仍可讀


if __name__ == "__main__":
    unittest.main()
