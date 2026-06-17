"""コマンドラインインターフェース。

使い方:
    python -m state_aware_rag.cli --mode socratic "your question"
    python -m state_aware_rag.cli --mode mcts --steps 5 --rollouts 10 "your question"
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from .config import load_config
from .pipeline import StateAwareRAG


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="State-Aware RAG (AWS Bedrock)")
    parser.add_argument("question", help="質問文")
    parser.add_argument("--mode", choices=["socratic", "mcts"], default=None)
    parser.add_argument("--steps", type=int, default=None, help="最大推論ステップ数")
    parser.add_argument("--rollouts", type=int, default=None, help="MCTS rollout 数")
    parser.add_argument("--json", action="store_true", help="JSON で出力")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # CLI 引数を環境変数経由で config に反映
    if args.mode:
        os.environ["SAR_MODE"] = args.mode
    if args.steps:
        os.environ["SAR_MAX_STEPS"] = str(args.steps)
    if args.rollouts:
        os.environ["SAR_MCTS_ROLLOUTS"] = str(args.rollouts)

    cfg = load_config()
    rag = StateAwareRAG(cfg)
    result = rag.answer(args.question)

    if args.json:
        print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))
    else:
        print(f"\n[mode] {result.mode}")
        print(f"[answer]\n{result.answer}\n")
        if args.verbose:
            print(f"[details]\n{json.dumps(result.details, ensure_ascii=False, indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
