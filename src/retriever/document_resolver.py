# -*- coding: utf-8 -*-
"""把向量索引命中安全地转换为正文与元数据。"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple


def _hit_value(hit: Any, key: str, default: Any = None) -> Any:
    getter = getattr(hit, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except Exception:
            return default
    if isinstance(hit, dict):
        return hit.get(key, default)
    return default


def resolve_milvus_document(
    hit: Any, mongo_document: Any
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """MongoDB 缺少记录时，回退到 Milvus output_fields 中保存的文本。"""
    unique_id = _hit_value(hit, "id") or _hit_value(hit, "unique_id") or ""
    if isinstance(mongo_document, dict):
        page_content = mongo_document.get("page_content") or _hit_value(hit, "text", "")
        metadata = mongo_document.get("metadata") or {}
    else:
        page_content = _hit_value(hit, "text", "")
        metadata = {"retrieval_source": "milvus_index"}

    if not page_content:
        return None
    if not isinstance(metadata, dict):
        metadata = {}
    else:
        metadata = dict(metadata)
    metadata.setdefault("unique_id", str(unique_id))
    return str(page_content), metadata
