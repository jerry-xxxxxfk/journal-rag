"""mcp_server.py：把「搜尋日誌」包成 MCP 工具，讓 Claude 可以直接呼叫。

MCP server 啟動後會用 stdin/stdout 跟 Claude 對話，所以：
  - 這個檔案裡「絕對不能用 print()」，否則會弄壞通訊（要輸出請用 stderr）。
  - 這個 server 只負責搜尋，不呼叫 Claude API，所以不花 API 費用。

手動啟動（通常由 Claude Code 自動啟動，不需要自己跑）：
    python mcp_server.py
"""

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

import os
from pathlib import Path

import ask

# 日誌所在資料夾（要和 ingest.py 建索引時用的資料夾一致）；可用環境變數 JOURNAL_DIR 指定
JOURNAL_DIR = Path(os.environ.get("JOURNAL_DIR", Path(__file__).parent / "sample_data"))
MAX_READ_CHARS = 8000  # 單次讀檔的字數上限

mcp = MCPServer(
    "journal-rag",
    instructions="搜尋使用者的學習日誌。當問題和使用者自己的學習、筆記、遇到的問題有關時使用。",
)


@mcp.tool()
def search_journal(question: str, top_k: int = 3) -> str:
    """在使用者的學習日誌中搜尋與問題最相關的段落。

    當使用者詢問自己學過什麼、遇到什麼困難、某天做了什麼時使用。
    回傳最相關的幾段日誌原文（含來源檔名、標題、相似度），請根據原文回答，
    不要自行編造日誌裡沒有的內容。

    Args:
        question: 要搜尋的問題或關鍵描述（中英文皆可）。
        top_k: 回傳幾段，1～5，預設 3。
    """
    top_k = max(1, min(5, top_k))
    try:
        hits = ask.retrieve(question, top_k=top_k)
    except FileNotFoundError:
        return "找不到索引檔，請先在專案資料夾執行：python ingest.py"
    return ask.format_hits(hits)


@mcp.tool()
def list_journals() -> str:
    """列出所有日誌檔名（檔名就是日期）。

    想知道有哪些日誌、最早或最晚是哪一天時使用。
    """
    names = [p.name for p in sorted(JOURNAL_DIR.glob("*.md"))]
    return "\n".join(names) if names else "沒有任何日誌。"


@mcp.tool()
def read_journal(filename: str) -> str:
    """讀取某一篇日誌的完整內容。

    需要比較、整理某一天的全部內容時使用。檔名請先用 list_journals 查詢。

    Args:
        filename: 日誌檔名，例如 2026-10-05.md。
    """
    # 安全：只允許讀「日誌資料夾裡真的存在的 .md 檔」。
    # 用「檔名對照表」而不是直接拼路徑，所以 "../.env" 之類的輸入一定找不到。
    allowed = {p.name: p for p in JOURNAL_DIR.glob("*.md")}
    path = allowed.get(filename)
    if path is None:
        # ToolError = 「預期中的失敗」：呼叫會被標記為失敗，而且這句訊息會原樣傳給 Claude，
        # 它才知道該改用 list_journals。（改丟一般錯誤的話，Claude 只會看到籠統的失敗訊息）
        raise ToolError(f"找不到日誌：{filename}。請先用 list_journals 查看有哪些檔案。")
    return path.read_text(encoding="utf-8")[:MAX_READ_CHARS]


if __name__ == "__main__":
    mcp.run(transport="stdio")
