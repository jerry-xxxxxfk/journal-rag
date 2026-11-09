"""ingest.py：把日誌「讀進來、切成小段、建好索引」。

用法：
    python ingest.py                 # 預設讀 sample_data/
    python ingest.py my_data         # 改讀你自己的真實日誌資料夾
"""

import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")  # 隱藏套件在舊版 macOS 上的無害警告

INDEX_FILE = Path(__file__).parent / "index.pkl"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# 模型放在專案資料夾內（已被 .gitignore 忽略），避免被系統清掉暫存後要重新下載
MODEL_CACHE = Path(__file__).parent / ".model_cache"

_model = None


def embed(texts: list[str]) -> np.ndarray:
    """把文字轉成「意思」的數字（embedding），並正規化成長度 1 的向量。"""
    global _model
    if _model is None:
        from fastembed import TextEmbedding

        _model = TextEmbedding(MODEL_NAME, cache_dir=str(MODEL_CACHE))
    vectors = np.array(list(_model.embed(texts)))
    # 正規化：長度都變成 1，之後「內積」就等於「夾角相似度」，值越大越像
    return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


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


def build_index(chunks: list[dict]) -> np.ndarray:
    """每一段文字 → 一個 384 維的向量，回傳 (段數, 384) 的矩陣。"""
    return embed([c["text"] for c in chunks])


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "sample_data"
    chunks = load_chunks(folder)
    if not chunks:
        sys.exit(f"在 {folder} 找不到任何 .md 檔")

    matrix = build_index(chunks)
    with open(INDEX_FILE, "wb") as f:
        pickle.dump({"matrix": matrix, "chunks": chunks, "model": MODEL_NAME}, f)

    print(f"完成：讀取 {folder}，切成 {len(chunks)} 段，索引存到 {INDEX_FILE.name}")


if __name__ == "__main__":
    main()
