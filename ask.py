"""ask.py：對你的日誌提問。

用法：
    python ask.py "我爬蟲遇到什麼問題？"
"""

import os
import pickle
import sys
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv

import ingest

# 預設讀專案內的 index.pkl；也可以用環境變數 JOURNAL_INDEX 指定別的索引檔
INDEX_FILE = Path(os.environ.get("JOURNAL_INDEX", Path(__file__).parent / "index.pkl"))
MODEL = "claude-haiku-5-5"
TOP_K = 3  # 每次找最相關的幾段


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """檢索：回傳最像問題的前 top_k 段，每段附上相似度分數。"""
    with open(INDEX_FILE, "rb") as f:
        index = pickle.load(f)

    question_vec = ingest.embed([question])[0]
    # 問題跟每一段做內積，分數越高「意思」越接近（向量已正規化，相當於夾角相似度）
    scores = index["matrix"] @ question_vec
    best = scores.argsort()[::-1][:top_k]
    return [{**index["chunks"][i], "score": float(scores[i])} for i in best]


def build_prompt(question: str, hits: list[dict]) -> str:
    """組合：把找到的段落和問題包成一份給 Claude 的說明。"""
    materials = "\n\n".join(
        f"【{h['source']}｜{h['heading']}】\n{h['text']}" for h in hits
    )
    return (
        "你是使用者的學習日誌助理。請只根據下面的「日誌片段」回答問題，"
        "並在答案中註明資料來自哪一篇。如果片段裡沒有答案，就直接說「日誌裡沒有相關紀錄」，"
        "不要自己編造。請用繁體中文，簡潔回答。\n\n"
        f"日誌片段：\n{materials}\n\n問題：{question}"
    )


def answer(question: str) -> tuple[str, list[dict]]:
    """生成：回傳 (Claude 的答案, 用到的片段)。"""
    hits = retrieve(question)
    if not hits:  # 完全沒找到相關內容，就不浪費錢呼叫 Claude
        return "日誌裡沒有相關紀錄。", []

    load_dotenv()  # 從 .env 讀取 ANTHROPIC_API_KEY
    client = Anthropic()
    response = client.messages.create(
        model=MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": build_prompt(question, hits)}],
    )
    # 回覆可能包含「思考區塊」等非文字內容，只取文字區塊
    text = "".join(b.text for b in response.content if b.type == "text")
    return text.strip() or "（模型沒有回傳文字，請再試一次）", hits


def main():
    if len(sys.argv) < 2:
        sys.exit('請提供問題，例如：python ask.py "我爬蟲遇到什麼問題？"')
    question = sys.argv[1]

    text, hits = answer(question)
    print(f"\n問題：{question}\n")
    print(f"答案：\n{text}\n")
    print("引用來源：")
    for h in hits:
        print(f"  - {h['source']}｜{h['heading']}（相似度 {h['score']:.2f}）")


if __name__ == "__main__":
    main()
