"""agent.py：透過 MCP 取用工具、自己決定「要用哪個工具、用幾次」的日誌 Agent。

和 ask.py 的差別：
  ask.py   固定流程：檢索一次 → 回答
  agent.py Claude 自己決定：先列日誌？搜尋？讀整篇？看資料夠了才回答（多步驟迴圈）

工具不是寫在這個檔案裡，而是由 mcp_server.py 提供：
  Agent 啟動時先連上 MCP server、問它「你有什麼工具」，再把清單交給 Claude；
  Claude 要用工具時，Agent 轉給 server 執行。所以 Claude Code 和這個 Agent 用的是同一組工具。

用法：
    python agent.py "比較我 10 月初和 10 月底的學習重點有什麼變化？"
"""

import asyncio
import os
import sys
from pathlib import Path

from anthropic import AsyncAnthropic
from dotenv import load_dotenv
from mcp import Client
from mcp.client.stdio import StdioServerParameters

import ask

ROOT = Path(__file__).parent
MAX_TURNS = 6  # 最多讓 Claude 想幾輪，避免卡住一直花錢

SYSTEM_PROMPT = (
    "你是使用者的學習日誌助理。你只能根據工具查到的日誌內容回答，不要編造。"
    "需要資料時，請自己選擇工具：想知道有哪些日誌用 list_journals；"
    "找特定主題用 search_journal；需要某一天的完整內容用 read_journal。"
    "資料足夠後再回答，並註明資料來自哪一篇日誌。請用繁體中文，簡潔回答。"
)


def default_server_params() -> StdioServerParameters:
    """怎麼啟動 MCP server：用同一個 Python 執行 mcp_server.py（它是我們的子程序）。"""
    return StdioServerParameters(
        command=sys.executable,
        args=[str(ROOT / "mcp_server.py")],
        env={**os.environ},  # 把 JOURNAL_DIR / JOURNAL_INDEX 等設定一起帶給 server
        cwd=str(ROOT),
    )


def to_claude_tools(mcp_tools) -> list[dict]:
    """把 MCP 的工具清單，轉成 Claude API 要求的格式（兩邊的欄位名稱不同）。"""
    return [
        {"name": t.name, "description": t.description or "", "input_schema": t.input_schema}
        for t in mcp_tools
    ]


async def call_mcp_tool(mcp: Client, name: str, tool_input: dict) -> tuple[str, bool]:
    """請 MCP server 執行工具，回傳 (結果文字, 是否出錯)。出錯時把訊息交給 Claude，讓它自己調整。"""
    try:
        result = await mcp.call_tool(name, tool_input)
    except Exception as e:  # noqa: BLE001  連線或協定出問題，不該讓整個 Agent 當掉
        return f"工具呼叫失敗：{type(e).__name__}: {e}", True
    text = "\n".join(b.text for b in result.content if b.type == "text")
    return text, bool(result.is_error)


async def run_agent(question: str, claude=None, max_turns: int = MAX_TURNS, on_step=None, server_params=None):
    """回傳 (最終答案, 工具呼叫紀錄)。claude 可以傳入假的物件來測試。"""
    if claude is None:
        load_dotenv()
        claude = AsyncAnthropic()

    messages = [{"role": "user", "content": question}]
    trace = []

    # 連上 MCP server（離開 with 區塊時會自動關掉子程序）
    async with Client(server_params or default_server_params()) as mcp:
        tools = to_claude_tools((await mcp.list_tools()).tools)

        for _ in range(max_turns):
            response = await claude.messages.create(
                model=ask.MODEL,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                tools=tools,
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
                output, is_error = await call_mcp_tool(mcp, block.name, dict(block.input))
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
        print(f"  🔧 透過 MCP 呼叫工具：{name}({args})")

    answer, trace = asyncio.run(run_agent(question, on_step=show))
    print(f"\n答案：\n{answer}\n")
    print(f"（共呼叫工具 {len(trace)} 次）")


if __name__ == "__main__":
    main()
