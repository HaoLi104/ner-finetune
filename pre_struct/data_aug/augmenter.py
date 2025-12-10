# augmenter.py
# -*- coding: utf-8 -*-
from __future__ import annotations

import json, random, re
from typing import Any, Dict, List

from structs import REPORT_STRUCTURE_MAP
try:
    from model_path_conf import DEFAULT_TOKENIZER_PATH  # type: ignore
except Exception as exc:
    raise ImportError(
        "DEFAULT_TOKENIZER_PATH must be defined in model_path_conf.py"
    ) from exc
from rag import TokenCounter
from llm_client import OpenAIFieldWiseLLM


class LLMPromptAugmenter:
    def __init__(
        self,
        tokenizer_name: str = DEFAULT_TOKENIZER_PATH,
        seed: int = 42,
        fieldwise_client: OpenAIFieldWiseLLM | None = None,
        compose_sep_prob: float = 0.5,
        compose_ocr_noise: bool = False,
        compose_paragraph_prob: float = 0.0,
        value_linebreak_prob: float = 0.0,
    ) -> None:
        self.tc = TokenCounter(tokenizer_name)
        random.seed(seed)
        if not fieldwise_client:
            raise ValueError("必须提供可用后端客户端")
        self.fieldwise_client = fieldwise_client
        self.compose_sep_prob = max(0.0, min(1.0, compose_sep_prob))
        self.compose_ocr_noise = bool(compose_ocr_noise)
        self.compose_paragraph_prob = max(0.0, min(1.0, compose_paragraph_prob))
        self.value_linebreak_prob = max(0.0, min(1.0, value_linebreak_prob))

    def _ocr_noise_via_llm(self, text: str) -> str:
        data = self.fieldwise_client._post_report(
            {
                "report": "【任务】模拟 OCR 轻中度字符噪声；不改语义/顺序；输出纯文本。\n【原文】\n"
                + text
            }
        )
        payload = data.get("report", data)
        if isinstance(payload, str):
            return payload.replace("```", "").strip()
        if isinstance(payload, dict) and isinstance(payload.get("report"), str):
            return payload["report"].strip()
        raise ValueError("后端返回格式不正确（OCR 噪声注入）")

    def _inject_value_linebreaks(self, s: str) -> str:
        if not s or self.value_linebreak_prob <= 0:
            return s
        parts = re.split(r"(?<=[。！？；.!?;])", s)
        out = []
        for i, seg in enumerate(parts):
            if not seg:
                continue
            out.append(seg)
            if i < len(parts) - 1 and random.random() < self.value_linebreak_prob:
                out.append("\n")
        return "".join(out)

    def compose_report(self, sample: Dict[str, Any], randomize: bool = True) -> str:
        lines = []
        title = str(sample.get("report_title", "")).strip()
        if title:
            lines.append(title)
        keys_map = REPORT_STRUCTURE_MAP.get(title)
        valid = []
        for k, v in sample.items():
            if (
                k in {"report", "report_title", "meta", "report_composed"}
                or not isinstance(k, str)
                or not k.strip()
                or v is None
            ):
                continue
            vs = str(v).replace("\r", "\n").strip()
            if vs:
                valid.append((k, vs))
        if keys_map:
            existed = {k for k, _ in valid}
            ordered = [k for k in keys_map if k in existed]
            rest = [k for k, _ in valid if k not in set(ordered)]
            iter_items = [(k, dict(valid)[k]) for k in (ordered + rest)]
        else:
            iter_items = valid
        for k, vs in iter_items:
            vs = self._inject_value_linebreaks(vs)
            if "检验数据" in str(k):
                lines.append(f" {vs}")
                continue
            sep = (
                random.choices(["：", ":"], [0.6, 0.4])[0]
                if (randomize and random.random() < self.compose_sep_prob)
                else ""
            )
            lines.append(f"{k}{sep if k else ' '}{vs}")
        report = "\n".join(lines)
        if self.compose_paragraph_prob > 0 and len(lines) > 2:
            out = []
            for i, ln in enumerate(lines):
                out.append(ln)
                if (
                    0 < i < (len(lines) - 1)
                    and random.random() < self.compose_paragraph_prob
                ):
                    out.append("")
            report = "\n".join(out)
        if self.compose_ocr_noise:
            report = self._ocr_noise_via_llm(report)
        return report

    def augment_once_with_client(
        self, sample: Dict[str, Any], client: OpenAIFieldWiseLLM
    ) -> Dict[str, Any]:
        data = client.augment_fields(sample)
        if not isinstance(data, dict):
            raise ValueError("后端返回非 dict（augment_fields）")
        data = {k: v for k, v in data.items() if k != "report"}
        if not data.get("report_title"):
            raise ValueError("后端增强缺少 'report_title'")
        data["report"] = self.compose_report(data)
        return data

    def augment_once(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        return self.augment_once_with_client(sample, self.fieldwise_client)

    def dataset_stats(self, records: List[Dict[str, Any]]) -> Dict[str, Any]:
        n, in_t, out_t, keyset, titles, key_cnt = len(records), 0, 0, set(), set(), []
        for r in records:
            t = r.get("report_title", "")
            if t:
                titles.add(t)
            in_t += self.tc.count(str(r.get("report", "")))
            ks = 0
            for k, v in r.items():
                keyset.add(k)
                if k in {"report", "report_title"}:
                    continue
                out_t += self.tc.count(
                    str(v)
                    if isinstance(v, (str, int, float))
                    else json.dumps(v, ensure_ascii=False)
                )
                ks += 1
            key_cnt.append(ks)
        return {
            "num_samples": n,
            "input_tokens": in_t,
            "output_tokens": out_t,
            "unique_key_count": len(keyset),
            "unique_titles": len(titles),
            "avg_keys_per_sample": round(sum(key_cnt) / max(1, n), 3),
        }

    @staticmethod
    def summarize_by_title(file_path: str) -> Dict[str, int]:
        from pathlib import Path
        import json

        data = json.loads(Path(file_path).read_text(encoding="utf-8"))
        from collections import Counter

        c = Counter([str(x.get("report_title", "")).strip() for x in data])
        c.pop("", None)
        return dict(c)
