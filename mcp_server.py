"""mcp_server.py：把「搜尋日誌」包成 MCP 工具，讓 Claude 可以直接呼叫。

MCP server 啟動後會用 stdin/stdout 跟 Claude 對話，所以：
  - 這個檔案裡「絕對不能用 print()」，否則會弄壞通訊（要輸出請用 stderr）。
  - 這個 server 只負責搜尋，不呼叫 Claude API，所以不花 API 費用。

手動啟動（通常由 Claude Code 自動啟動，不需要自己跑）：
    python mcp_server.py
"""

from mcp.server.mcpserver import MCPServer

import ask

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


if __name__ == "__main__":
    mcp.run(transport="stdio")
