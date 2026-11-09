"""ask.py：對你的日誌提問。

用法：
    python ask.py "我爬蟲遇到什麼問題？"
"""

import pickle
import sys
from pathlib import Path

from anthropic import Anthropic
from dotenv import load_dotenv
from sklearn.metrics.pairwise import linear_kernel

INDEX_FILE = Path(__file__).parent / "index.pkl"
MODEL = "claude-haiku-5-5"
TOP_K = 3  # 每次找最相關的幾段


def retrieve(question: str, top_k: int = TOP_K) -> list[dict]:
    """檢索：回傳最像問題的前 top_k 段，每段附上相似度分數。"""
    with open(INDEX_FILE, "rb") as f:
        index = pickle.load(f)

    question_vec = index["vectorizer"].transform([question])
    # 問題跟每一段做相似度比較，分數越高越像（0 = 完全沒重疊）
    scores = linear_kernel(question_vec, index["matrix"]).ravel()
    best = scores.argsort()[::-1][:top_k]
    return [{**index["chunks"][i], "score": float(scores[i])} for i in best if scores[i] > 0]


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
    return response.content[0].text, hits


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
