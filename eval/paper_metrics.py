"""論文の評価指標（Sub-EM と Acc）。

- Sub-EM (Substring Exact Match): 正解文字列が応答に部分一致するか（厳密な字句一致）
- Acc (Accuracy): LLM-as-a-Judge による意味的一致（論文は Claude 3.7 Sonnet を使用）

RAGAS のような追加依存なしで計算できる、論文準拠の基本指標。
"""
from __future__ import annotations


def sub_em(prediction: str, answers: list[str]) -> bool:
    """正解候補のいずれかが応答に（大文字小文字を無視して）部分一致するか。"""
    p = (prediction or "").lower()
    return any(a.lower() in p for a in answers if a)


def aggregate(rows: list[dict], key: str) -> float:
    """bool/数値メトリクスの平均（%）。"""
    vals = [float(r[key]) for r in rows if r.get(key) is not None]
    return round(100.0 * sum(vals) / len(vals), 1) if vals else 0.0
