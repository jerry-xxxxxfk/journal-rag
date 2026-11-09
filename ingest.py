"""ingest.py：把日誌「讀進來、切成小段、建好索引」。

用法：
    python ingest.py                 # 預設讀 sample_data/
    python ingest.py my_data         # 改讀你自己的真實日誌資料夾
"""

import pickle
import sys
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer

INDEX_FILE = Path(__file__).parent / "index.pkl"


def split_into_chunks(path: Path) -> list[dict]:
    """把一篇 .md 按「## 標題」切成多段，每段記住出處。"""
    chunks = []
    heading = "（開頭）"
    lines = []

    def save():
        text = "\n".join(lines).strip()
        if text:
            chunks.append(
                {"source": path.name, "heading": heading, "text": text}
            )

    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):  # 遇到新的二級標題：先存上一段，再開新的一段
            save()
            heading = line[3:].strip()
            lines = []
        elif line.startswith("# "):  # 一級標題是整篇的日期標題，不當成內容
            continue
        else:
            lines.append(line)
    save()  # 別忘了最後一段
    return chunks


def load_chunks(folder: Path) -> list[dict]:
    """讀取資料夾裡所有 .md，回傳所有段落。"""
    chunks = []
    for path in sorted(folder.glob("*.md")):
        chunks.extend(split_into_chunks(path))
    return chunks


def build_index(chunks: list[dict]):
    """把每一段文字轉成一串數字（TF-IDF），之後才能比較「誰跟問題最像」。"""
    # analyzer="char"：以「字元片段」為單位，中文不用先斷詞也能比對
    # ngram_range=(1, 2)：同時看單字和連續兩個字（例如「爬蟲」「蟲被」）
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform(c["text"] for c in chunks)
    return vectorizer, matrix


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "sample_data"
    chunks = load_chunks(folder)
    if not chunks:
        sys.exit(f"在 {folder} 找不到任何 .md 檔")

    vectorizer, matrix = build_index(chunks)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": matrix, "chunks": chunks}, f)

    print(f"完成：讀取 {folder}，切成 {len(chunks)} 段，索引存到 {INDEX_FILE.name}")


if __name__ == "__main__":
    main()
