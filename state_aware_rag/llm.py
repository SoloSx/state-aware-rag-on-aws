"""Amazon Bedrock を介した LLM クライアント。

論文の Generator / Extractor / Judge をすべて Bedrock の Converse API 経由で呼ぶ。
論文の SGLang + LiteLLM プロキシ構成に相当する役割を Bedrock が担う。
"""
from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import boto3
from botocore.config import Config as BotoConfig

from .config import BedrockConfig

logger = logging.getLogger(__name__)


class BedrockLLM:
    """Bedrock Converse API のシンプルなラッパー。

    role に応じて temperature を切り替える（論文 Appendix B.3）:
      - generator: temperature=1.0
      - extractor / judge: temperature=0.1
    """

    def __init__(self, cfg: BedrockConfig):
        self.cfg = cfg
        boto_cfg = BotoConfig(
            retries={"max_attempts": 6, "mode": "adaptive"},
            read_timeout=300,
            connect_timeout=10,
        )
        self.client = boto3.client("bedrock-runtime", region_name=cfg.region, config=boto_cfg)

    # -- 低レベル呼び出し ---------------------------------------------------
    def _invoke(self, model_id: str, prompt: str, temperature: float, max_tokens: int | None = None) -> str:
        # 注意: 一部の新しい Claude（Sonnet 4.5 等）は temperature と topP の
        # 同時指定を拒否する。論文は両方指定だが、ここでは temperature を優先し
        # top_p は送らない（temperature をステップ役割ごとに変える方が本質的）。
        body = {
            "modelId": model_id,
            "messages": [{"role": "user", "content": [{"text": prompt}]}],
            "inferenceConfig": {
                "temperature": temperature,
                "maxTokens": max_tokens or self.cfg.max_tokens,
            },
        }
        for attempt in range(5):
            try:
                resp = self.client.converse(**body)
                return resp["output"]["message"]["content"][0]["text"]
            except self.client.exceptions.ThrottlingException:
                wait = 2 ** attempt
                logger.warning("Bedrock throttled; retrying in %ss", wait)
                time.sleep(wait)
        raise RuntimeError(f"Bedrock invocation failed after retries: {model_id}")

    # -- role 別ヘルパー ----------------------------------------------------
    def generate(self, prompt: str, **kw) -> str:
        """Generator（中間回答・サブ質問・最終回答）。temperature=1.0。"""
        return self._invoke(self.cfg.generator_model_id, prompt, self.cfg.generator_temperature, **kw)

    def extract(self, prompt: str, **kw) -> str:
        """Extractor（filter / consolidate / memory update）。temperature=0.1。"""
        return self._invoke(self.cfg.extractor_model_id, prompt, self.cfg.other_temperature, **kw)

    def judge(self, prompt: str, **kw) -> str:
        """Judge（報酬・評価）。temperature=0.1。"""
        return self._invoke(self.cfg.judge_model_id, prompt, self.cfg.other_temperature, **kw)

    # -- JSON 出力のパース --------------------------------------------------
    @staticmethod
    def parse_json(text: str) -> dict[str, Any]:
        """LLM 出力から最初の JSON オブジェクトを頑健に取り出す。

        Claude が thinking や ```json フェンスを付けても拾えるようにする。
        """
        # ```json ... ``` フェンスを優先
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if fence:
            candidate = fence.group(1)
        else:
            # 最初の '{' から対応する '}' までを括弧の深さで抽出
            start = text.find("{")
            if start == -1:
                return {}
            depth = 0
            end = -1
            for i in range(start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        end = i + 1
                        break
            candidate = text[start:end] if end != -1 else text[start:]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            logger.warning("JSON parse failed; returning empty dict. raw=%.200s", text)
            return {}
