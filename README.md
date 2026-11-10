# journal-rag

一個最小的 RAG（檢索增強生成）工具：對自己的學習日誌提問，Claude 只根據日誌內容回答，並附上引用來源。

這個 repo 包含「RAG → MCP → Agent」三個串在一起的小作品，目的是用最小的程式量，把每一步都做清楚、講得出原理：

| 檔案 | 作品 | 做什麼 |
| :-- | :-- | :-- |
| `ingest.py` + `ask.py` | RAG | 切段、建 embedding 索引、檢索後讓 Claude 回答並附出處 |
| `mcp_server.py` | MCP | 把「搜尋日誌」包成 MCP 工具，讓 Claude Code 等 MCP 用戶端直接呼叫 |
| `agent.py` | Agent | MCP 用戶端：向 `mcp_server.py` 取得工具清單，由 Claude 自己決定用哪個工具、用幾次的多步驟迴圈 |

## 運作方式

```
sample_data/*.md
   │  ingest.py
   ├─ ① 讀檔
   ├─ ② 按「## 標題」切成小段（每段記住出處）
   └─ ③ 每段轉成 embedding 向量（384 維，代表「意思」）→ index.pkl

提問時（ask.py）
   問題 → ④ 轉成向量，找出意思最接近的 3 段 → ⑤ 連同問題交給 Claude → 答案 + 引用來源
```

- **檢索**（③④）使用本機的多語言 embedding 模型 `paraphrase-multilingual-MiniLM-L12-v2`（透過 `fastembed`，約 0.2 GB，不需要 GPU，不花錢）。
- **生成**（⑤）呼叫 Claude Haiku，是唯一用到 Anthropic API 的地方。
- 提示詞要求 Claude「只根據片段回答，沒有就說沒有紀錄」，降低亂編的機率。

## 使用方式

需要 **Python 3.10 以上**（`mcp` 套件的要求）與 [Anthropic API key](https://console.anthropic.com/)。

```bash
# 1. 建立環境並安裝套件
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. 設定 API key（.env 已被 .gitignore 忽略，不會上傳）
echo "ANTHROPIC_API_KEY=你的金鑰" > .env

# 3. 建立索引（第一次執行會下載 embedding 模型，約 1 分鐘）
python ingest.py            # 預設讀 sample_data/

# 4. 提問
python ask.py "網站不讓我進去怎麼辦"
```

範例輸出（日誌原文寫的是「403 被擋」，問題用了完全不同的說法）：

```
答案：
根據 2026-10-05.md（遇到的困難）：
- 網站回傳 403 錯誤，代表伺服器把你的程式當成機器人擋掉。
- 解法是加上瀏覽器的 User-Agent 標頭，並且每次請求之間暫停幾秒。
...

引用來源：
  - 2026-10-05.md｜遇到的困難（相似度 0.49）
  - 2026-10-05.md｜心得（相似度 0.40）
  - 2026-10-05.md｜今天學了什麼（相似度 0.37）
```

`sample_data/` 內是虛構的範例日誌。想用自己的日誌，放進 `my_data/`（已被忽略，不會上傳），再執行 `python ingest.py my_data`。

## MCP server

`mcp_server.py` 把 `ask.retrieve()` 包成一個 MCP 工具 `search_journal`。MCP 是讓 AI 連接外部工具的統一規格：server 負責「自我介紹」（有什麼工具、何時使用）和「接受呼叫」，由 AI 自己決定要不要用。

- 這個 server 只做搜尋，**不呼叫 Claude API**，所以不產生 API 費用。
- 注意：stdio 模式下 stdout 是通訊管道，程式中不能使用 `print()`。
- 測試會真的啟動 server，用 MCP client 連線、列出工具並呼叫（`tests/test_mcp_server.py`）。

### 接上 Claude Code

```bash
# 路徑請換成你自己的（要用 venv 裡的 python，才找得到套件）
claude mcp add journal-rag -- /path/to/journal-rag/venv/bin/python /path/to/journal-rag/mcp_server.py
claude mcp list        # 應顯示 journal-rag: ✔ Connected
```

註冊後重新啟動 Claude Code，就可以直接用自然語言問。實測：問「我日誌裡有寫過 403 嗎？」，Claude Code 呼叫 `search_journal`，回傳 `2026-10-05.md｜遇到的困難`（相似度 0.44，排第 1）的原文。

## Agent

`agent.py` 與 `ask.py` 的差別：`ask.py` 是固定流程（檢索一次 → 回答）；Agent 則由 Claude 自己決定下一步，直到資料足夠才回答。

```
問題 → Claude 思考 ──要工具──► 程式執行工具 ──結果──┐
          ▲                                        │
          └────────────────────────────────────────┘
          │ 資料足夠
          ▼
        最終答案
```

**工具不寫在 `agent.py` 裡，而是由 `mcp_server.py` 提供**：Agent 啟動時以子程序方式連上 MCP server、詢問有哪些工具，轉成 Claude API 的格式交給 Claude；Claude 要用工具時，Agent 再透過 `call_tool()` 轉給 server 執行。所以 Claude Code 和這個 Agent 用的是同一組工具，只需維護一份。

```
agent.py（MCP 用戶端）──MCP 協定──► mcp_server.py
   │ list_tools → 工具清單 → 交給 Claude            ├─ list_journals（列出檔名）
   └ Claude 要工具 → call_tool → 結果還給 Claude     ├─ search_journal（語意搜尋）
                                                    └─ read_journal（讀整篇）
```

```bash
python agent.py "比較我 10 月初和 10 月底的學習重點有什麼變化？"
```

實際執行時（工具呼叫全部經過 MCP），Claude 先呼叫 `list_journals`，再逐篇 `read_journal`（共 5 次工具呼叫），最後整理出比較。

安全與成本設計：
- 迴圈最多 6 輪，避免卡住一直花錢。
- `read_journal`（在 MCP server 端）只能讀日誌資料夾內實際存在的 `.md` 檔（以檔名對照表查找，而非直接拼接路徑），`../.env` 這類輸入一律拒絕，並有測試驗證。
- 工具失敗時 server 用 `ToolError` 回報，錯誤訊息（含「請改用 list_journals」的提示）會原樣傳給 Claude；若丟一般例外，Claude 只會看到籠統的失敗訊息。
- 工具出錯時會標記 `is_error` 回報給 Claude，讓它自行調整。

## 測試

```bash
python -m unittest discover -s tests -t . -v
```

共 21 個測試，涵蓋切段、檢索、MCP server（端到端，含路徑穿越）與 Agent（用「照劇本回應的假 Claude」驅動迴圈，但工具走真的 MCP server），全部不呼叫 Claude API。檢索測試的標準是「正確的來源出現在前 3 名」，因為 `ask.py` 會把前 3 段一起交給 Claude。

## 設計取捨與演進

### v0：TF-IDF（git tag `v0-tfidf`）

第一版用 TF-IDF（以 1～2 個字元為單位）做檢索：輕量、不用下載模型、原理容易說明。但它**只比對字面，不懂意思**。實測 5 個問題：

| 問題 | TF-IDF | embedding（目前版本） |
| :-- | :-- | :-- |
| 爬蟲被 403 擋掉怎麼辦（字面重疊） | ✅ 第 1 名 | ✅ 第 1 名 |
| 網站不讓我進去怎麼辦（同義說法） | ❌ 正確段落沒進前 2 名 | ✅ 第 1 名 |
| what did I learn about loops（英文問中文） | ❌ 找到錯的篇章 | ✅ 第 1 名 |
| 我找工作準備得怎麼樣（口語） | ✅ | ✅ |
| 我怎麼讓AI接上外部工具 | ✅ 第 1 名 | ⚠️ 第 2 名（0.47 vs 0.49），仍在前 3 名內 |

TF-IDF 還有一個問題：相似度分數無法當信心指標，錯誤結果的分數可能比正確結果還高。

### v1：embedding（目前）

換成 embedding 後，兩個失敗案例都修好了，只有一題從第 1 名退到第 2 名。**注意：** 我是在看到這個結果之後，才把測試標準從「第 1 名」改成「前 3 名」，這個標準符合實際用法（前 3 段都會給 Claude），但並非事先訂好的。

## 已知限制

- 樣本只有 4 篇、12 段，結果好不代表在大量資料上也一樣；沒有做系統性的評估（只有上面 5 個手選問題）。
- 使用的是最小的多語言模型（0.22 GB），更大的模型（如 `paraphrase-multilingual-mpnet-base-v2`，1.0 GB）可能排名更準，尚未測試。
- 沒有設定「相似度太低就不回答」的門檻；完全無關的問題仍會取到 3 段，靠提示詞讓 Claude 回答「沒有相關紀錄」。
- 索引存成單一 pickle 檔，資料量大時應改用向量資料庫。
- Agent 只用 4 篇日誌示範，Claude 直接讀完全部，沒有展現到 `search_journal` 的價值；資料量大時才會看出差別，尚未驗證。
- Agent 每次執行都要啟動一個 MCP server 子程序並載入 embedding 模型，啟動會多花幾秒；多一層通訊也多一個可能出錯的環節。

## 下一步

- [x] 把 TF-IDF 換成 embedding
- [x] 把 `retrieve()` 包成 MCP server
- [x] 能自己選工具的多步驟 Agent
- [x] 讓 Agent 改成透過 MCP 連線取用工具
- [ ] 用更多、更長的日誌做系統性評估
