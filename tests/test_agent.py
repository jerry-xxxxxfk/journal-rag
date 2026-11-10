"""Agent 的測試：假 Claude 照劇本回應，但工具走「真的 MCP server」（端到端）。

執行方式（在專案根目錄）：
    python -m unittest discover -s tests -t . -v

不呼叫 Claude API，不花錢。工具本身（含路徑穿越防護）的細部測試在 test_mcp_server.py。
"""

import os
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from mcp import Client

import agent
import ingest

SAMPLE_DIR = Path(__file__).parent.parent / "sample_data"


def tool_use(tool_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)


def reply(stop_reason, *blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


class FakeClaude:
    """按照劇本依序回傳預先寫好的回應，並記錄每次收到的對話與工具清單。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.tools_seen = None
        self.messages = SimpleNamespace(create=self._create)

    async def _create(self, **kwargs):
        self.tools_seen = kwargs["tools"]
        self.calls.append(list(kwargs["messages"]))  # 記錄當時的對話（淺拷貝）
        return self.script[len(self.calls) - 1]


class TestAgentOverMCP(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        # 暫存索引（給 search_journal 用）；日誌資料夾用 sample_data（假資料）
        cls._tmp = tempfile.TemporaryDirectory()
        chunks = ingest.load_chunks(SAMPLE_DIR)
        index_path = Path(cls._tmp.name) / "index.pkl"
        with open(index_path, "wb") as f:
            pickle.dump({"matrix": ingest.build_index(chunks), "chunks": chunks, "model": ingest.MODEL_NAME}, f)

        params = agent.default_server_params()
        params.env = {**os.environ, "JOURNAL_INDEX": str(index_path), "JOURNAL_DIR": str(SAMPLE_DIR)}
        cls.params = params

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    async def test_tools_come_from_mcp_server_in_claude_format(self):
        async with Client(self.params) as mcp:
            tools = agent.to_claude_tools((await mcp.list_tools()).tools)
        self.assertEqual({t["name"] for t in tools}, {"search_journal", "list_journals", "read_journal"})
        for t in tools:
            self.assertEqual(t["input_schema"]["type"], "object")
            self.assertTrue(t["description"])

    async def test_multi_step_loop_runs_tools_through_mcp(self):
        claude = FakeClaude([
            reply("tool_use", tool_use("t1", "list_journals", {})),
            reply("tool_use", tool_use("t2", "read_journal", {"filename": "2026-10-01.md"})),
            reply("end_turn", SimpleNamespace(type="text", text="最終答案")),
        ])
        steps = []
        answer, trace = await agent.run_agent(
            "比較日誌", claude=claude, server_params=self.params, on_step=lambda n, i: steps.append(n)
        )

        self.assertEqual(answer, "最終答案")
        self.assertEqual([t["tool"] for t in trace], ["list_journals", "read_journal"])
        self.assertEqual(steps, ["list_journals", "read_journal"])
        self.assertEqual(len(claude.calls), 3)
        # Claude 看到的工具清單，來自 MCP server
        self.assertEqual({t["name"] for t in claude.tools_seen}, {"search_journal", "list_journals", "read_journal"})

        # 第 2 次呼叫 Claude 時，對話裡必須帶著「第 1 個工具的真實結果」
        block = claude.calls[1][-1]["content"][0]
        self.assertEqual(block["type"], "tool_result")
        self.assertEqual(block["tool_use_id"], "t1")
        self.assertFalse(block["is_error"])
        self.assertIn("2026-10-05.md", block["content"])

    async def test_search_journal_goes_through_mcp(self):
        claude = FakeClaude([
            reply("tool_use", tool_use("t1", "search_journal", {"question": "爬蟲被 403 擋掉怎麼辦", "top_k": 2})),
            reply("end_turn", SimpleNamespace(type="text", text="好")),
        ])
        await agent.run_agent("403？", claude=claude, server_params=self.params)
        block = claude.calls[1][-1]["content"][0]
        self.assertFalse(block["is_error"])
        self.assertIn("2026-10-05.md", block["content"])
        self.assertEqual(block["content"].count("【"), 2)

    async def test_tool_error_reaches_claude_with_hint_and_no_secret(self):
        claude = FakeClaude([
            reply("tool_use", tool_use("t1", "read_journal", {"filename": "../.env"})),
            reply("end_turn", SimpleNamespace(type="text", text="好的")),
        ])
        await agent.run_agent("讀機密", claude=claude, server_params=self.params)
        block = claude.calls[1][-1]["content"][0]
        self.assertTrue(block["is_error"])
        self.assertIn("list_journals", block["content"])  # 提示有傳到 Claude
        self.assertNotIn("ANTHROPIC", block["content"])

    async def test_unknown_tool_is_reported_as_error(self):
        claude = FakeClaude([
            reply("tool_use", tool_use("t1", "delete_everything", {})),
            reply("end_turn", SimpleNamespace(type="text", text="好的")),
        ])
        await agent.run_agent("亂叫工具", claude=claude, server_params=self.params)
        block = claude.calls[1][-1]["content"][0]
        self.assertTrue(block["is_error"])

    async def test_loop_stops_after_max_turns(self):
        endless = reply("tool_use", tool_use("t", "list_journals", {}))
        claude = FakeClaude([endless] * 10)
        answer, trace = await agent.run_agent("永遠不停", claude=claude, max_turns=3, server_params=self.params)
        self.assertIn("超過 3 輪", answer)
        self.assertEqual(len(claude.calls), 3)  # 沒有超過上限
        self.assertEqual(len(trace), 3)


if __name__ == "__main__":
    unittest.main()
