"""Agent 的測試：用「照劇本回應的假 Claude」驗證迴圈與安全防護，不呼叫 API、不花錢。

執行方式（在專案根目錄）：
    python -m unittest discover -s tests -t . -v
"""

import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import agent
import ask
import ingest

SAMPLE_DIR = Path(__file__).parent.parent / "sample_data"


def tool_use(tool_id, name, tool_input):
    return SimpleNamespace(type="tool_use", id=tool_id, name=name, input=tool_input)


def reply(stop_reason, *blocks):
    return SimpleNamespace(stop_reason=stop_reason, content=list(blocks))


class FakeClaude:
    """按照劇本依序回傳預先寫好的回應，並記錄每次收到的對話內容。"""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(list(kwargs["messages"]))  # 記錄當時的對話（淺拷貝）
        return self.script[len(self.calls) - 1]


class TestTools(unittest.TestCase):
    def test_list_journals(self):
        names = agent.list_journals().splitlines()
        self.assertEqual(len(names), 4)
        self.assertIn("2026-10-05.md", names)

    def test_read_journal_returns_content(self):
        self.assertIn("403", agent.read_journal("2026-10-05.md"))

    def test_read_journal_blocks_path_traversal(self):
        # 在日誌資料夾「外面」放一個機密檔，確認 Agent 讀不到它
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "secret.txt").write_text("SECRET=abc123", encoding="utf-8")
            journals = Path(tmp) / "journals"
            journals.mkdir()
            (journals / "2026-01-01.md").write_text("# 一般日誌", encoding="utf-8")

            original = agent.JOURNAL_DIR
            agent.JOURNAL_DIR = journals
            try:
                for evil in ["../secret.txt", "../../etc/passwd", "/etc/passwd", "", "secret.txt"]:
                    result, is_error = agent.run_tool("read_journal", {"filename": evil})
                    self.assertTrue(is_error, evil)
                    self.assertNotIn("SECRET", result, evil)
                    self.assertIn("找不到日誌", result, evil)
                self.assertIn("一般日誌", agent.read_journal("2026-01-01.md"))  # 正常檔案仍可讀
            finally:
                agent.JOURNAL_DIR = original

    def test_unknown_tool_is_reported_as_error(self):
        text, is_error = agent.run_tool("delete_everything", {})
        self.assertTrue(is_error)
        self.assertIn("沒有這個工具", text)

    def test_bad_arguments_do_not_crash(self):
        text, is_error = agent.run_tool("read_journal", {"wrong_arg": 1})
        self.assertTrue(is_error)
        self.assertIn("工具執行失敗", text)


class TestSearchTool(unittest.TestCase):
    def test_search_journal_tool_returns_hits(self):
        with tempfile.TemporaryDirectory() as tmp:
            chunks = ingest.load_chunks(SAMPLE_DIR)
            index_path = Path(tmp) / "index.pkl"
            with open(index_path, "wb") as f:
                pickle.dump({"matrix": ingest.build_index(chunks), "chunks": chunks, "model": ingest.MODEL_NAME}, f)
            original = ask.INDEX_FILE
            ask.INDEX_FILE = index_path
            try:
                result = agent.search_journal("爬蟲被 403 擋掉怎麼辦", top_k=2)
            finally:
                ask.INDEX_FILE = original
        self.assertIn("2026-10-05.md", result)
        self.assertEqual(result.count("【"), 2)


class TestAgentLoop(unittest.TestCase):
    def test_multi_step_loop_uses_tools_then_answers(self):
        client = FakeClaude([
            reply("tool_use", tool_use("t1", "list_journals", {})),
            reply("tool_use", tool_use("t2", "read_journal", {"filename": "2026-10-01.md"})),
            reply("end_turn", SimpleNamespace(type="text", text="最終答案")),
        ])
        steps = []
        answer, trace = agent.run_agent("比較日誌", client=client, on_step=lambda n, i: steps.append(n))

        self.assertEqual(answer, "最終答案")
        self.assertEqual([t["tool"] for t in trace], ["list_journals", "read_journal"])
        self.assertEqual(steps, ["list_journals", "read_journal"])
        self.assertEqual(len(client.calls), 3)

        # 第 2 次呼叫 Claude 時，對話裡必須帶著「第 1 個工具的結果」
        last_user = client.calls[1][-1]
        self.assertEqual(last_user["role"], "user")
        self.assertEqual(last_user["content"][0]["type"], "tool_result")
        self.assertEqual(last_user["content"][0]["tool_use_id"], "t1")
        self.assertIn("2026-10-05.md", last_user["content"][0]["content"])

    def test_loop_stops_after_max_turns(self):
        endless = reply("tool_use", tool_use("t", "list_journals", {}))
        client = FakeClaude([endless] * 10)
        answer, trace = agent.run_agent("永遠不停", client=client, max_turns=3)
        self.assertIn("超過 3 輪", answer)
        self.assertEqual(len(client.calls), 3)  # 沒有超過上限
        self.assertEqual(len(trace), 3)

    def test_tool_error_is_passed_back_to_claude(self):
        client = FakeClaude([
            reply("tool_use", tool_use("t1", "read_journal", {"filename": "../.env"})),
            reply("end_turn", SimpleNamespace(type="text", text="好的")),
        ])
        agent.run_agent("讀機密", client=client)
        result_block = client.calls[1][-1]["content"][0]
        self.assertTrue(result_block["is_error"])
        self.assertNotIn("ANTHROPIC", result_block["content"])


if __name__ == "__main__":
    unittest.main()
