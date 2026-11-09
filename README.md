# journal-rag

一個最小的 RAG（檢索增強生成）工具：對自己的學習日誌提問，Claude 只根據日誌內容回答，並附上引用來源。

這是「RAG → MCP → Agent」三部曲的第 1 個作品。目的是用最小的程式量，把 RAG 的每一步都做清楚、講得出原理。

## 運作方式

```
sample_data/*.md
   │  ingest.py
   ├─ ① 讀檔
   ├─ ② 按「## 標題」切成小段（每段記住出處）
   └─ ③ 每段轉成 TF-IDF 向量 → index.pkl

提問時（ask.py）
   問題 → ④ 找出最相似的 3 段 → ⑤ 連同問題交給 Claude → 答案 + 引用來源
```

- **檢索**（④）完全在本機執行，不花錢。
- **生成**（⑤）呼叫 Claude Haiku，是唯一用到 API 的地方。
- 提示詞要求 Claude「只根據片段回答，沒有就說沒有紀錄」，降低亂編的機率。

## 使用方式

需要 Python 3.9 以上與 [Anthropic API key](https://console.anthropic.com/)。

```bash
# 1. 建立環境並安裝套件
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 2. 設定 API key（.env 已被 .gitignore 忽略，不會上傳）
echo "ANTHROPIC_API_KEY=你的金鑰" > .env

# 3. 建立索引（預設讀 sample_data/，也可指定自己的資料夾）
python ingest.py

# 4. 提問
python ask.py "我爬蟲遇到什麼問題？"
```

範例輸出：

```
答案：
根據 2026-10-05.md，你在爬蟲時遇到的問題有：
1. 403 錯誤：網站把你當成機器人擋掉
2. 部分內容由 JavaScript 載入，單純抓 HTML 看不到

引用來源：
  - 2026-10-05.md｜今天學了什麼（相似度 0.15）
  - 2026-10-05.md｜遇到的困難（相似度 0.14）
```

`sample_data/` 內是虛構的範例日誌。想用自己的日誌，放進 `my_data/`（已被忽略，不會上傳），再執行 `python ingest.py my_data`。

## 測試

```bash
python -m unittest discover -s tests -t . -v
```

測試涵蓋切段與檢索，不呼叫 API。

## 已知限制（刻意保留的取捨）

檢索使用 TF-IDF（以 1～2 個字元為單位），優點是輕量、免下載模型、原理容易說明；缺點是**只比對字面，不懂意思**：

| 問題 | 日誌原文 | 結果 |
| :-- | :-- | :-- |
| 爬蟲被 403 擋掉怎麼辦 | 403 被擋 | ✅ 找到 |
| 網站不讓我進去怎麼辦 | 403 被擋 | ❌ 找不到（同義說法，沒有共同用字） |
| what did I learn about loops | 中文日誌 | ❌ 找不到（跨語言） |

這兩個失敗案例已寫成測試（標記為 expected failure），之後升級後應該會轉為通過。

另外，TF-IDF 的相似度分數不能當作信心指標：錯誤結果的分數可能比正確結果還高。

## 下一步

- [ ] 把 TF-IDF 換成 embedding（比對意思，改善同義詞與跨語言）
- [ ] 第 2 個作品：把 `retrieve()` 包成 MCP server，讓 Claude 直接呼叫
- [ ] 第 3 個作品：能自己選工具的多步驟 Agent
