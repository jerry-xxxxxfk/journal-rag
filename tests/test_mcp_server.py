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


if __name__ == "__main__":
    unittest.main()
