"""agent.py：會自己決定「要用哪個工具、用幾次」的日誌 Agent。

和 ask.py 的差別：
  ask.py   固定流程：檢索一次 → 回答
  agent.py Claude 自己決定：先列日誌？搜尋？讀整篇？看資料夠了才回答（多步驟迴圈）

用法：
    python agent.py "比較我 10 月初和 10 月底的學習重點有什麼變化？"
"""

import os
import sys
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

import ask

# 日誌所在資料夾（要和 ingest.py 建索引時用的資料夾一致）
JOURNAL_DIR = Path(os.environ.get("JOURNAL_DIR", Path(__file__).parent / "sample_data"))
MAX_TURNS = 6  # 最多讓 Claude 想幾輪，避免卡住一直花錢
MAX_READ_CHARS = 8000  # 單次讀檔的字數上限

SYSTEM_PROMPT = (
    "你是使用者的學習日誌助理。你只能根據工具查到的日誌內容回答，不要編造。"
    "需要資料時，請自己選擇工具：想知道有哪些日誌用 list_journals；"
    "找特定主題用 search_journal；需要某一天的完整內容用 read_journal。"
    "資料足夠後再回答，並註明資料來自哪一篇日誌。請用繁體中文，簡潔回答。"
)

TOOLS = [
    {
        "name": "list_journals",
        "description": "列出所有日誌檔名（檔名就是日期）。想知道有哪些日誌、最早或最晚是哪一天時使用。",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "search_journal",
        "description": "用意思搜尋日誌，回傳最相關的幾個段落。想找某個主題或問題在哪篇日誌時使用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "要搜尋的問題或主題"},
                "top_k": {"type": "integer", "description": "回傳幾段，1～5，預設 3"},
            },
            "required": ["question"],
        },
    },
    {
        "name": "read_journal",
        "description": "讀取某一篇日誌的完整內容。需要比較、整理某一天的全部內容時使用。",
        "input_schema": {
            "type": "object",
            "properties": {
                "filename": {"type": "string", "description": "日誌檔名，例如 2026-10-05.md（先用 list_journals 查）"},
            },
            "required": ["filename"],
        },
    },
]


# ---------- 工具的實際執行內容 ----------

def list_journals() -> str:
    names = [p.name for p in sorted(JOURNAL_DIR.glob("*.md"))]
    return "\n".join(names) if names else "沒有任何日誌。"


def search_journal(question: str, top_k: int = 3) -> str:
    top_k = max(1, min(5, int(top_k)))
    try:
        return ask.format_hits(ask.retrieve(question, top_k=top_k))
    except FileNotFoundError:
        return "找不到索引檔，請先執行：python ingest.py"


def read_journal(filename: str) -> str:
    # 安全：只允許讀「日誌資料夾裡真的存在的 .md 檔」。
    # 用「檔名對照表」而不是直接拼路徑，所以 "../.env" 之類的輸入一定找不到。
    allowed = {p.name: p for p in JOURNAL_DIR.glob("*.md")}
    path = allowed.get(filename)
    if path is None:
        # 丟出錯誤 → run_tool 會標記 is_error=True，Claude 就知道這次呼叫失敗、該換個做法
        raise FileNotFoundError(f"找不到日誌：{filename}。請先用 list_journals 查看有哪些檔案。")
    return path.read_text(encoding="utf-8")[:MAX_READ_CHARS]


TOOL_FUNCTIONS = {
    "list_journals": list_journals,
    "search_journal": search_journal,
    "read_journal": read_journal,
}


def run_tool(name: str, tool_input: dict) -> tuple[str, bool]:
    """執行工具，回傳 (結果文字, 是否出錯)。出錯時把錯誤告訴 Claude，讓它自己調整。"""
    func = TOOL_FUNCTIONS.get(name)
    if func is None:
        return f"沒有這個工具：{name}", True
    try:
        return func(**tool_input), False
    except Exception as e:  # noqa: BLE001  工具出錯不該讓整個 Agent 當掉
        return f"工具執行失敗：{type(e).__name__}: {e}", True


# ---------- Agent 迴圈 ----------

def run_agent(question: str, client=None, max_turns: int = MAX_TURNS, on_step=None):
    """回傳 (最終答案, 工具呼叫紀錄)。client 可以傳入假的物件來測試。"""
    if client is None:
        load_dotenv()
        client = Anthropic()

    messages = [{"role": "user", "content": question}]
    trace = []

    for _ in range(max_turns):
        response = client.messages.create(
            model=ask.MODEL,
            max_tokens=2000,
            system=SYSTEM_PROMPT,
            tools=TOOLS,
            messages=messages,
        )

        if response.stop_reason != "tool_use":
            # Claude 認為資料夠了，這就是最終答案
            text = "".join(b.text for b in response.content if b.type == "text").strip()
            return text or "（模型沒有回傳文字）", trace

        # Claude 要求用工具：先把它這一輪的話原樣記進對話，再把工具結果還給它
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            output, is_error = run_tool(block.name, block.input)
            trace.append({"tool": block.name, "input": dict(block.input), "error": is_error})
            if on_step:
                on_step(block.name, dict(block.input))
            results.append(
                {"type": "tool_result", "tool_use_id": block.id, "content": output, "is_error": is_error}
            )
        messages.append({"role": "user", "content": results})

    return f"（超過 {max_turns} 輪仍未得出答案，已停止。）", trace


def main():
    if len(sys.argv) < 2:
        sys.exit('請提供問題，例如：python agent.py "比較我 10 月初和 10 月底的學習重點"')
    question = sys.argv[1]
    print(f"\n問題：{question}\n")

    def show(name, tool_input):
        args = ", ".join(f"{k}={v!r}" for k, v in tool_input.items())
        print(f"  🔧 呼叫工具：{name}({args})")

    answer, trace = run_agent(question, on_step=show)
    print(f"\n答案：\n{answer}\n")
    print(f"（共呼叫工具 {len(trace)} 次）")


if __name__ == "__main__":
    main()
